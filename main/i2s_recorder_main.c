#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "driver/sdmmc_host.h"  // 使用 SDMMC 主機驅動

#include "esp_check.h"
#include "esp_vfs_fat.h"
#include "esp_timer.h"
#include "sdmmc_cmd.h"

// ================= I2S 麥克風腳位定義 =================
#define I2S_BCLK_IO     GPIO_NUM_26   
#define I2S_WS_IO       GPIO_NUM_25   
#define I2S_DIN_IO      GPIO_NUM_33   

// ================= 錄音參數設定 =================
#define SAMPLE_RATE             44100
#define RECORD_TIME_SEC         2
#define WAV_BITS_PER_SAMPLE     16
#define WAV_CHANNELS            1

#define DMA_DESC_NUM            32
#define DMA_FRAME_NUM           512
#define I2S_READ_BYTES          4092

#define WRITE_BUFFER_SIZE       32768

// DBG PIN 避開了 GPIO 12，改用 GPIO 27 (GPIO 12 是開機引腳，容易引發問題)
#define DBG_PIN GPIO_NUM_27 
#define DBG_SD_PIN GPIO_NUM_18
#define CAPTURE_PIN GPIO_NUM_5

#define AUDIO_BLOCK_SAMPLES   1024
#define AUDIO_BLOCK_COUNT     16

typedef struct {
    int16_t samples[AUDIO_BLOCK_SAMPLES];
    size_t  sample_count;
} audio_block_t;

static QueueHandle_t q_free_blocks;
static QueueHandle_t q_full_blocks;
static audio_block_t g_blocks[AUDIO_BLOCK_COUNT];

static const char *TAG = "I2S Record";

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

static i2s_chan_handle_t rx_handle = NULL;
static sdmmc_card_t *sd_card = NULL;

int overflow_count = 0; 

static void audio_block_pool_init(void)
{
    q_free_blocks = xQueueCreate(AUDIO_BLOCK_COUNT, sizeof(audio_block_t *));
    q_full_blocks = xQueueCreate(AUDIO_BLOCK_COUNT, sizeof(audio_block_t *));

    for (int i = 0; i < AUDIO_BLOCK_COUNT; i++) {
        audio_block_t *p = &g_blocks[i];
        xQueueSend(q_free_blocks, &p, portMAX_DELAY);
    }
}

static void dbg_pin_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = 1ULL << DBG_PIN ,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = 0,
        .pull_down_en = 0,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    gpio_set_level(DBG_PIN , 0);
}

static void fill_wav_header(wav_header_t *hdr, uint32_t data_size)
{
    memcpy(hdr->riff, "RIFF", 4);
    memcpy(hdr->wave, "WAVE", 4);
    memcpy(hdr->fmt_, "fmt ", 4);
    memcpy(hdr->data, "data", 4);

    hdr->file_size_minus_8 = data_size + 36;
    hdr->fmt_chunk_size = 16;
    hdr->audio_format = 1;
    hdr->num_channels = WAV_CHANNELS;
    hdr->sample_rate = SAMPLE_RATE;
    hdr->bits_per_sample = WAV_BITS_PER_SAMPLE;
    hdr->byte_rate = SAMPLE_RATE * WAV_CHANNELS * (WAV_BITS_PER_SAMPLE / 8); 
    hdr->block_align = WAV_CHANNELS * (WAV_BITS_PER_SAMPLE / 8);
    hdr->data_size = data_size;
}

// 🚀 針對 TTGO-T8 專屬改寫：SDMMC 1-Bit 高速模式初始化
static esp_err_t sd_card_init(void)
{
    esp_err_t ret;
    
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024,
    };

    ESP_LOGI(TAG, "Initializing SD card using SDMMC 1-bit mode (TTGO-T8 Optimized)");

    // 1. 設定 SDMMC 主機，開啟高速模式 (40MHz)
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED; 

    // 2. 設定 Slot
    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    
    // ⭐️ 關鍵修改：TTGO-T8 只有接出 D0，因此設定為 1-bit 模式
    // 驅動層會自動綁定: CMD(GPIO15), CLK(GPIO14), D0(GPIO2)
    slot_config.width = 1; 
    
    // 啟用內部上拉電阻以增加穩定性
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    // 3. 掛載 SD 卡
    ret = esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot_config, &mount_config, &sd_card);
    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount filesystem.");
        } else {
            ESP_LOGE(TAG, "Failed to initialize the card (%s).", esp_err_to_name(ret));
        }
        return ret;
    }

    sdmmc_card_print_info(stdout, sd_card);
    return ESP_OK;
}

static void sd_card_deinit(void)
{
    if (sd_card) {
        esp_vfs_fat_sdcard_unmount("/sdcard", sd_card);
        sd_card = NULL;
        ESP_LOGI(TAG, "Card unmounted");
    }
}

static bool IRAM_ATTR i2s_rx_overflow_callback(i2s_chan_handle_t handle, i2s_event_data_t *event, void *user_ctx)
{
    gpio_set_level(DBG_PIN, 1);
    overflow_count++; 
    return false;
}

static esp_err_t i2s_init_equivalent_to_ino(void)
{
    i2s_chan_config_t chan_cfg = {
        .id = I2S_NUM_0,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = DMA_DESC_NUM,
        .dma_frame_num = DMA_FRAME_NUM,
        .auto_clear = false,
    };

    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, NULL, &rx_handle), "I2S", "i2s_new_channel failed");

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_IO,
            .ws = I2S_WS_IO,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_DIN_IO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    std_cfg.slot_cfg.ws_pol = false;
    std_cfg.slot_cfg.bit_shift = true;
    std_cfg.slot_cfg.msb_right = false;

    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(rx_handle, &std_cfg), "I2S", "i2s_channel_init_std_mode failed");

    i2s_event_callbacks_t cbs = {
        .on_recv = NULL,
        .on_recv_q_ovf = i2s_rx_overflow_callback,
        .on_sent = NULL,
        .on_send_q_ovf = NULL
    };
    ESP_RETURN_ON_ERROR(i2s_channel_register_event_callback(rx_handle, &cbs, NULL), "I2S", "Register cb failed");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(rx_handle), "I2S", "i2s_channel_enable failed");

    return ESP_OK;
}

static void i2s_deinit(void)
{
    if (rx_handle) {
        i2s_channel_disable(rx_handle);
        i2s_del_channel(rx_handle);
        rx_handle = NULL;
    }
}

static esp_err_t record(const char *path, uint32_t seconds)
{
    size_t target_samples = SAMPLE_RATE * seconds;
    size_t pcm_bytes = target_samples * sizeof(int16_t);
    size_t raw_read_bytes = I2S_READ_BYTES;

    int32_t *i2s_read_buffer = (int32_t *)malloc(raw_read_bytes);
    int16_t *full_pcm_buffer = (int16_t *)malloc(pcm_bytes);

    if (!i2s_read_buffer || !full_pcm_buffer) {
        ESP_LOGE(TAG, "malloc failed");
        free(i2s_read_buffer);
        free(full_pcm_buffer);
        return ESP_FAIL;
    }

    size_t total_samples = 0;
    size_t bytes_read = 0;

    gpio_set_level(DBG_PIN, 1);

    while (total_samples < target_samples) {
        esp_err_t ret = i2s_channel_read(rx_handle, i2s_read_buffer, raw_read_bytes, &bytes_read, 100);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "i2s read failed: %d", ret);
            break;
        }

        int samples_in_chunk = bytes_read / sizeof(int32_t);

        for (int i = 0; i < samples_in_chunk && total_samples < target_samples; i++) {
            int32_t raw = i2s_read_buffer[i];
            int16_t pcm = (int16_t)(raw >> 16);
            full_pcm_buffer[total_samples++] = pcm;
        }
    }

    gpio_set_level(DBG_PIN, 0);

    FILE *f = fopen(path, "wb");
    if (!f) {
        ESP_LOGE(TAG, "open file failed");
        free(i2s_read_buffer);
        free(full_pcm_buffer);
        return ESP_FAIL;
    }

    wav_header_t hdr;
    fill_wav_header(&hdr, total_samples * sizeof(int16_t));
    fwrite(&hdr, 1, sizeof(hdr), f);
    // 這裡運用了 fwrite 的整包寫入，Buffer Size 取決於 full_pcm_buffer，寫入效率極高
    fwrite(full_pcm_buffer, sizeof(int16_t), total_samples, f);
    fclose(f);

    ESP_LOGI(TAG, "RAM record done: %u samples, %u bytes",
             (unsigned)total_samples,
             (unsigned)(total_samples * sizeof(int16_t)));

    free(i2s_read_buffer);
    free(full_pcm_buffer);
    return ESP_OK;
}

void app_main(void)
{
    dbg_pin_init();
    
    // 初始化 SD 卡 (TTGO-T8 專用 SDMMC 1-Bit 模式)
    if (sd_card_init() != ESP_OK) {
        printf("SD Card initialization failed.\n");
        return;
    }
    printf("SD Card initialized successfully.\n");

    // 初始化 I2S
    if (i2s_init_equivalent_to_ino() != ESP_OK) {
        printf("I2S initialization failed.\n");
        sd_card_deinit();
        return;
    }
    printf("I2S initialized\n");

    // 開始錄音並寫入 SD 卡
    if (record("/sdcard/audio.wav", RECORD_TIME_SEC) != ESP_OK) {
        printf("Recording failed.\n");
    } else {
        printf("Recording finished: /sdcard/audio.wav\n");
    }

    i2s_deinit();
    sd_card_deinit();
}