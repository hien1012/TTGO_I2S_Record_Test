#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "driver/spi_common.h"

#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "esp_heap_caps.h"

#include "sdmmc_cmd.h"

// ================= 硬體腳位定義 =================
#define I2S_BCLK_IO     GPIO_NUM_26   // SCK / BCLK
#define I2S_WS_IO       GPIO_NUM_25   // WS / LRCLK
#define I2S_DIN_IO      GPIO_NUM_33   // SD

#define SD_MISO_PIN     GPIO_NUM_2
#define SD_MOSI_PIN     GPIO_NUM_15
#define SD_CLK_PIN      GPIO_NUM_14
#define SD_CS_PIN       GPIO_NUM_13

// Debug GPIO (選用)
#define DBG_REC_PIN     GPIO_NUM_12   // capture task 進行中
#define DBG_SD_PIN      GPIO_NUM_18   // writer task 正在 fwrite

// ================= 錄音參數 =================
#define SAMPLE_RATE             16000
#define RECORD_TIME_SEC         5
#define WAV_BITS_PER_SAMPLE     16
#define WAV_CHANNELS            1

// I2S 讀取參數
#define DMA_DESC_NUM            8
#define DMA_FRAME_NUM           512
#define I2S_READ_BYTES          8096

// 32-bit raw -> 16-bit PCM 的縮位量
// 可依實測改成 14 / 11 / 16
#define PCM_SHIFT_BITS          16

// ================= 雙任務 / Block Pool 參數 =================
#define AUDIO_BLOCK_SAMPLES     1024
#define AUDIO_BLOCK_COUNT       16
#define FILE_IO_BUFFER_SIZE     32768

static const char *TAG = "I2S_REC";
static const char *g_record_path = "/sdcard/audio.wav";

typedef struct __attribute__((packed)) {
    char riff[4];
    uint32_t file_size_minus_8;
    char wave[4];
    char fmt_[4];
    uint32_t fmt_chunk_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
} wav_header_t;

typedef struct {
    size_t sample_count;
    int16_t samples[AUDIO_BLOCK_SAMPLES]; // 每一個存1024個 int16_t
} audio_block_t;

typedef struct {
    const char *name;
    TaskFunction_t entry;
    uint32_t stack_size;
    UBaseType_t priority;
    BaseType_t core_id;
    void *arg;
    TaskHandle_t *handle_out;
} task_plan_t;

static i2s_chan_handle_t rx_handle = NULL;
static sdmmc_card_t *sd_card = NULL;

static QueueHandle_t q_free_blocks = NULL;
static QueueHandle_t q_full_blocks = NULL;

static TaskHandle_t g_capture_task = NULL;
static TaskHandle_t g_writer_task = NULL;
static TaskHandle_t g_main_task = NULL;

static volatile uint32_t overflow_count = 0;
static volatile bool capture_done = false;
static volatile bool writer_done = false;
static size_t g_total_samples_written = 0;

// 固定記憶體池：避免 runtime malloc 失敗
static audio_block_t g_audio_blocks[AUDIO_BLOCK_COUNT]; // 開16個 block, 每一個block 有 sample count / 1024個 int16_t 
static int32_t g_i2s_read_buffer[I2S_READ_BYTES / sizeof(int32_t)];
static char g_file_io_buffer[FILE_IO_BUFFER_SIZE];

static void dbg_pin_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << DBG_REC_PIN) | (1ULL << DBG_SD_PIN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    gpio_set_level(DBG_REC_PIN, 0);
    gpio_set_level(DBG_SD_PIN, 0);
}

//========================
//  Fill WAV Header infomation 
//======================== 
static void fill_wav_header(wav_header_t *hdr, uint32_t data_size)
{
    memcpy(hdr->riff, "RIFF", 4);
    memcpy(hdr->wave, "WAVE", 4);
    memcpy(hdr->fmt_, "fmt ", 4);
    memcpy(hdr->data, "data", 4);

    hdr->file_size_minus_8 = data_size + 36;
    hdr->fmt_chunk_size = 16;
    hdr->audio_format = 1; // PCM
    hdr->num_channels = WAV_CHANNELS;
    hdr->sample_rate = SAMPLE_RATE;
    hdr->bits_per_sample = WAV_BITS_PER_SAMPLE;
    hdr->byte_rate = SAMPLE_RATE * WAV_CHANNELS * (WAV_BITS_PER_SAMPLE / 8);
    hdr->block_align = WAV_CHANNELS * (WAV_BITS_PER_SAMPLE / 8);
    hdr->data_size = data_size;
}

static esp_err_t sd_card_init(void)
{
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024,
    };

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = SPI2_HOST;

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = SD_MOSI_PIN,
        .miso_io_num = SD_MISO_PIN,
        .sclk_io_num = SD_CLK_PIN,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4096,
    };

    ESP_RETURN_ON_ERROR(spi_bus_initialize(host.slot, &bus_cfg, SPI_DMA_CH_AUTO), TAG, "spi_bus_initialize failed");

    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = SD_CS_PIN;
    slot_config.host_id = host.slot;

    return esp_vfs_fat_sdspi_mount("/sdcard", &host, &slot_config, &mount_config, &sd_card);
}

static void sd_card_deinit(void)
{
    if (sd_card) {
        esp_vfs_fat_sdcard_unmount("/sdcard", sd_card);
        sd_card = NULL;
    }
    spi_bus_free(SPI2_HOST);
}

//========================
//  偵測DMA 是否 overflow 
//======================== 
static bool IRAM_ATTR i2s_rx_overflow_callback(i2s_chan_handle_t handle, i2s_event_data_t *event, void *user_ctx)
{
    (void)handle;
    (void)event;
    (void)user_ctx;
    overflow_count++;
    return false;
}

//========================
//  初始化 I2S  
//======================== 
static esp_err_t i2s_init(void)
{
    i2s_chan_config_t chan_cfg = {
        .id = I2S_NUM_0,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = DMA_DESC_NUM,
        .dma_frame_num = DMA_FRAME_NUM,
        .auto_clear = true,
    };

    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, NULL, &rx_handle), TAG, "i2s_new_channel failed");

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_IO,
            .ws   = I2S_WS_IO,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_DIN_IO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    // 與你目前 ino 對齊的設定：Philips / Left / Mono
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    std_cfg.slot_cfg.ws_pol = false;
    std_cfg.slot_cfg.bit_shift = true;
    std_cfg.slot_cfg.msb_right = false;

    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(rx_handle, &std_cfg), TAG, "i2s_channel_init_std_mode failed");

    i2s_event_callbacks_t cbs = {
        .on_recv = NULL,
        .on_recv_q_ovf = i2s_rx_overflow_callback,
        .on_sent = NULL,
        .on_send_q_ovf = NULL,
    };
    ESP_RETURN_ON_ERROR(i2s_channel_register_event_callback(rx_handle, &cbs, NULL), TAG, "register cb failed");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(rx_handle), TAG, "i2s_channel_enable failed");

    return ESP_OK;
}

//========================
//  Deinitialize I2S  
//======================== 
static void i2s_deinit(void)
{
    if (rx_handle) {
        i2s_channel_disable(rx_handle);
        i2s_del_channel(rx_handle);
        rx_handle = NULL;
    }
}


//============================================
// Open a file in SDcard to write the WAV file 
//============================================ 
static FILE *wav_open_placeholder(const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        ESP_LOGE(TAG, "Cannot open file: %s", path);
        return NULL;
    }

    setvbuf(f, g_file_io_buffer, _IOFBF, sizeof(g_file_io_buffer));

    wav_header_t hdr = {0};
    fill_wav_header(&hdr, 0);
    fwrite(&hdr, 1, sizeof(hdr), f);
    return f;
}

//==========================================
//  Write the size of the recoreded to WAV file
//========================================== 
static void wav_finalize(FILE *f, size_t total_samples_written)
{
    wav_header_t hdr = {0};
    size_t data_bytes = total_samples_written * sizeof(int16_t);

    fill_wav_header(&hdr, data_bytes);
    fseek(f, 0, SEEK_SET);
    fwrite(&hdr, 1, sizeof(hdr), f);
    fclose(f);
}

//==============================================
//  Initialize QUEUE for "Audio Record Pipeline" 
//============================================== 
static esp_err_t audio_pipeline_init(void)
{
    q_free_blocks = xQueueCreate(AUDIO_BLOCK_COUNT, sizeof(audio_block_t *));
    q_full_blocks = xQueueCreate(AUDIO_BLOCK_COUNT + 1, sizeof(audio_block_t *));

    if (!q_free_blocks || !q_full_blocks) {
        ESP_LOGE(TAG, "Queue create failed");
        return ESP_FAIL;
    }

    for (int i = 0; i < AUDIO_BLOCK_COUNT; i++) {
        audio_block_t *blk = &g_audio_blocks[i];
        blk->sample_count = 0;
        xQueueSend(q_free_blocks, &blk, portMAX_DELAY);
    }

    return ESP_OK;
}

//==============================================
//  Delete QUEUE for "Audio Record Pipeline" 
//============================================== 
static void audio_pipeline_deinit(void)
{
    if (q_free_blocks) {
        vQueueDelete(q_free_blocks);
        q_free_blocks = NULL;
    }
    if (q_full_blocks) {
        vQueueDelete(q_full_blocks);
        q_full_blocks = NULL;
    }
}

//==============================================================================
//  @brief  sound task (fill data to free block), send it to full block by QUEUE 
//============================================================================== 
static void capture_task(void *arg)
{
    (void)arg;

    size_t target_samples = SAMPLE_RATE * RECORD_TIME_SEC;
    size_t total_samples_captured = 0;
    size_t bytes_read = 0;
    int64_t t0 = esp_timer_get_time();

    audio_block_t *blk = NULL;
    xQueueReceive(q_free_blocks, &blk, portMAX_DELAY);
    blk->sample_count = 0;

    capture_done = false;
    gpio_set_level(DBG_REC_PIN, 1);
    ESP_LOGI(TAG, "capture_task start, target_samples=%u", (unsigned)target_samples);

    while (total_samples_captured < target_samples) {
        esp_err_t ret = i2s_channel_read(rx_handle,             /* data is from i2s buffer -> dma -> ram      */
                                         g_i2s_read_buffer,     /* i2s_channel_read copy data from dma -> ram */
                                         I2S_READ_BYTES,        /*  I2S_READ_BYTE at one time */
                                         &bytes_read,
                                         100);

        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "i2s_channel_read failed: %d", ret);
            break;
        }

        int n = bytes_read / sizeof(int32_t);
        for (int i = 0; i < n && total_samples_captured < target_samples; i++) {
            int32_t raw = g_i2s_read_buffer[i];
            int16_t pcm = (int16_t)(raw >> PCM_SHIFT_BITS);

            blk->samples[blk->sample_count++] = pcm;
            total_samples_captured++;

            if (blk->sample_count >= AUDIO_BLOCK_SAMPLES) {
                xQueueSend(q_full_blocks, &blk, portMAX_DELAY);
                xQueueReceive(q_free_blocks, &blk, portMAX_DELAY);
                blk->sample_count = 0;
            }
        }
    }

    if (blk && blk->sample_count > 0) {
        xQueueSend(q_full_blocks, &blk, portMAX_DELAY);
    }

    blk = NULL; // sentinel
    xQueueSend(q_full_blocks, &blk, portMAX_DELAY);

    gpio_set_level(DBG_REC_PIN, 0);
    capture_done = true;

    double elapsed_s = (esp_timer_get_time() - t0) / 1000000.0;
    ESP_LOGI(TAG, "capture_task done: %u samples, elapsed=%.3f s", (unsigned)total_samples_captured, elapsed_s);
    vTaskDelete(NULL);
}


/**
 * @brief sound data to SDCard, 清空所接收到的 fullblock, 將這個block 傳遞給 capture task
 */
static void writer_task(void *arg)
{
    const char *path = (const char *)arg;
    FILE *f = wav_open_placeholder(path);
    if (!f) {
        xTaskNotifyGive(g_main_task);
        vTaskDelete(NULL);
        return;
    }

    writer_done = false;
    g_total_samples_written = 0;

    ESP_LOGI(TAG, "writer_task start: %s", path);

    while (1) {
        audio_block_t *blk = NULL;
        xQueueReceive(q_full_blocks, &blk, portMAX_DELAY);

        if (blk == NULL) {
            break;
        }

        gpio_set_level(DBG_SD_PIN, 1);
        size_t written = fwrite(blk->samples, sizeof(int16_t), blk->sample_count, f);
        gpio_set_level(DBG_SD_PIN, 0);

        if (written != blk->sample_count) {
            ESP_LOGE(TAG, "fwrite short: expect=%u actual=%u",
                     (unsigned)blk->sample_count,
                     (unsigned)written);
        }

        g_total_samples_written += written;
        blk->sample_count = 0;
        xQueueSend(q_free_blocks, &blk, portMAX_DELAY);
    }

    wav_finalize(f, g_total_samples_written);
    writer_done = true;

    ESP_LOGI(TAG, "writer_task done: %u samples, duration=%.3f s",
             (unsigned)g_total_samples_written,
             (double)g_total_samples_written / SAMPLE_RATE);

    xTaskNotifyGive(g_main_task);
    vTaskDelete(NULL);
}

/**
 * @brief 一個array, 所使用到的task的資訊設定struct
 * @return NONE
 */
static const task_plan_t g_task_plan[] = {
    {
        .name = "capture_task",
        .entry = capture_task,
        .stack_size = 4096,
        .priority = 10,
        .core_id = 1,
        .arg = NULL,
        .handle_out = &g_capture_task,
    },
    {
        .name = "writer_task",
        .entry = writer_task,
        .stack_size = 6144,
        .priority = 6,
        .core_id = 0,
        .arg = (void *)"/sdcard/audio.wav",
        .handle_out = &g_writer_task,
    },
};

void app_main(void)
{
    g_main_task = xTaskGetCurrentTaskHandle();

    ESP_LOGI(TAG, "Free internal heap: %u", (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT));
#ifdef CONFIG_SPIRAM
    ESP_LOGI(TAG, "Free PSRAM: %u", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#endif

    dbg_pin_init();

    if (sd_card_init() != ESP_OK) {
        ESP_LOGE(TAG, "SD Card initialization failed");
        return;
    }
    ESP_LOGI(TAG, "SD Card initialized successfully");

    if (i2s_init() != ESP_OK) {
        ESP_LOGE(TAG, "I2S initialization failed");
        sd_card_deinit();
        return;
    }
    ESP_LOGI(TAG, "I2S initialized");

    if (audio_pipeline_init() != ESP_OK) {
        ESP_LOGE(TAG, "audio_pipeline_init failed");
        i2s_deinit();
        sd_card_deinit();
        return;
    }

    for (size_t i = 0; i < sizeof(g_task_plan) / sizeof(g_task_plan[0]); i++) {
        const task_plan_t *t = &g_task_plan[i];
        BaseType_t ok = xTaskCreatePinnedToCore(
            t->entry,
            t->name,
            t->stack_size,
            t->arg,
            t->priority,
            t->handle_out,
            t->core_id
        );

        if (ok != pdPASS) {
            ESP_LOGE(TAG, "Create task failed: %s", t->name);
            audio_pipeline_deinit();
            i2s_deinit();
            sd_card_deinit();
            return;
        }
    }

    // 等 writer_task 收到 sentinel 並完成 WAV 封尾
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    ESP_LOGI(TAG, "Recording finished: %s", g_record_path);
    ESP_LOGI(TAG, "capture_done=%d writer_done=%d overflow_count=%u total_samples_written=%u",
             capture_done, writer_done, (unsigned)overflow_count, (unsigned)g_total_samples_written);

    audio_pipeline_deinit();
    i2s_deinit();
    sd_card_deinit();
}
