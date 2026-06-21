#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "driver/i2s_std.h"
#include "driver/gpio.h"

#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"

#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"

// I2S microphone pins
#define I2S_BCLK_IO             GPIO_NUM_26
#define I2S_WS_IO               GPIO_NUM_25
#define I2S_DIN_IO              GPIO_NUM_33

// Button: pressed = LOW, internal pull-up enabled
#define BTN_PIN                 GPIO_NUM_0

// Debug GPIOs
#define DBG_REC_PIN             GPIO_NUM_12
#define DBG_SD_PIN              GPIO_NUM_18

#define SAMPLE_RATE             16000
#define RECORD_TIME_SEC         6
#define WAV_BITS_PER_SAMPLE     16
#define WAV_CHANNELS            1

#define DMA_DESC_NUM            8
#define DMA_FRAME_NUM           512
#define I2S_READ_BYTES          8096
#define I2S_FLUSH_READS         4
#define I2S_READ_TIMEOUT_MS     1000

// The current microphone stream is 32-bit; keep the same conversion as
// i2s_recorder_dual_task.c so recordings are directly comparable.
#define PCM_SHIFT_BITS          16
#define PCM_GAIN                30

#define BTN_DEBOUNCE_MS         50
#define BTN_POST_RELEASE_MS     300
#define FILE_IO_BUFFER_SIZE     32768

static const char *TAG = "REC_SINGLE";

typedef struct __attribute__((packed)) {
    char     riff[4];
    uint32_t file_size_minus_8;
    char     wave[4];
    char     fmt_[4];
    uint32_t fmt_chunk_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char     data[4];
    uint32_t data_size;
} wav_header_t;

static i2s_chan_handle_t rx_handle = NULL;
static sdmmc_card_t *sd_card = NULL;

static SemaphoreHandle_t g_btn_pressed_sem = NULL;
static SemaphoreHandle_t g_record_start_sem = NULL;

static volatile uint32_t overflow_count = 0;
static int32_t g_i2s_read_buffer[I2S_READ_BYTES / sizeof(int32_t)];
static char g_file_io_buffer[FILE_IO_BUFFER_SIZE];
static char g_wav_path[32];

static void fill_wav_header(wav_header_t *hdr, uint32_t data_size)
{
    memcpy(hdr->riff, "RIFF", 4);
    memcpy(hdr->wave, "WAVE", 4);
    memcpy(hdr->fmt_, "fmt ", 4);
    memcpy(hdr->data, "data", 4);

    hdr->file_size_minus_8 = data_size + 36;
    hdr->fmt_chunk_size    = 16;
    hdr->audio_format      = 1;
    hdr->num_channels      = WAV_CHANNELS;
    hdr->sample_rate       = SAMPLE_RATE;
    hdr->bits_per_sample   = WAV_BITS_PER_SAMPLE;
    hdr->byte_rate         = SAMPLE_RATE * WAV_CHANNELS * (WAV_BITS_PER_SAMPLE / 8);
    hdr->block_align       = WAV_CHANNELS * (WAV_BITS_PER_SAMPLE / 8);
    hdr->data_size         = data_size;
}

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

static void IRAM_ATTR btn_isr_handler(void *arg)
{
    (void)arg;
    BaseType_t higher_prio_woken = pdFALSE;
    xSemaphoreGiveFromISR(g_btn_pressed_sem, &higher_prio_woken);
    portYIELD_FROM_ISR(higher_prio_woken);
}

static void btn_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << BTN_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE,
    };
    gpio_config(&io_conf);
    gpio_install_isr_service(0);
    gpio_isr_handler_add(BTN_PIN, btn_isr_handler, NULL);
}

static void button_task(void *arg)
{
    (void)arg;

    while (1) {
        xSemaphoreTake(g_btn_pressed_sem, portMAX_DELAY);

        vTaskDelay(pdMS_TO_TICKS(BTN_DEBOUNCE_MS));
        if (gpio_get_level(BTN_PIN) != 0) {
            continue;
        }

        while (gpio_get_level(BTN_PIN) == 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }

        vTaskDelay(pdMS_TO_TICKS(BTN_POST_RELEASE_MS));
        xSemaphoreTake(g_btn_pressed_sem, 0);

        ESP_LOGI(TAG, "Button confirmed released -> start single-buffer recording");
        xSemaphoreGive(g_record_start_sem);
    }
}

static int get_next_file_index(void)
{
    int idx = 1;
    char path[32];

    while (1) {
        snprintf(path, sizeof(path), "/sdcard/%d.wav", idx);
        FILE *f = fopen(path, "rb");
        if (!f) {
            break;
        }
        fclose(f);
        idx++;
    }
    return idx;
}

static esp_err_t sd_card_init(void)
{
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024,
    };

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.flags = SDMMC_HOST_FLAG_1BIT;
    host.max_freq_khz = 20000;

    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 1;
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    return esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot_config, &mount_config, &sd_card);
}

static void sd_card_deinit(void)
{
    if (sd_card) {
        esp_vfs_fat_sdcard_unmount("/sdcard", sd_card);
        sd_card = NULL;
    }
}

static bool IRAM_ATTR i2s_rx_overflow_callback(i2s_chan_handle_t handle,
                                                i2s_event_data_t *event,
                                                void *user_ctx)
{
    (void)handle;
    (void)event;
    (void)user_ctx;
    overflow_count++;
    return false;
}

static esp_err_t i2s_init(void)
{
    i2s_chan_config_t chan_cfg = {
        .id = I2S_NUM_0,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = DMA_DESC_NUM,
        .dma_frame_num = DMA_FRAME_NUM,
        .auto_clear = true,
    };
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, NULL, &rx_handle),
                        TAG, "i2s_new_channel failed");

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_MONO),
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

    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(rx_handle, &std_cfg),
                        TAG, "i2s_channel_init_std_mode failed");

    i2s_event_callbacks_t cbs = {
        .on_recv = NULL,
        .on_recv_q_ovf = i2s_rx_overflow_callback,
        .on_sent = NULL,
        .on_send_q_ovf = NULL,
    };
    ESP_RETURN_ON_ERROR(i2s_channel_register_event_callback(rx_handle, &cbs, NULL),
                        TAG, "register cb failed");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(rx_handle),
                        TAG, "i2s_channel_enable failed");
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

static esp_err_t record_to_ram(int16_t *pcm_buffer,
                               size_t target_samples,
                               size_t *samples_recorded)
{
    size_t total_samples = 0;
    size_t bytes_read = 0;

    for (int flush = 0; flush < I2S_FLUSH_READS; flush++) {
        esp_err_t ret = i2s_channel_read(rx_handle,
                                         g_i2s_read_buffer,
                                         I2S_READ_BYTES,
                                         &bytes_read,
                                         pdMS_TO_TICKS(I2S_READ_TIMEOUT_MS));
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "I2S flush read %d failed: %s", flush, esp_err_to_name(ret));
        }
    }

    gpio_set_level(DBG_REC_PIN, 1);
    int64_t t0 = esp_timer_get_time();
    ESP_LOGI(TAG, "Capture start: target=%u samples", (unsigned)target_samples);

    while (total_samples < target_samples) {
        esp_err_t ret = i2s_channel_read(rx_handle,
                                         g_i2s_read_buffer,
                                         I2S_READ_BYTES,
                                         &bytes_read,
                                         pdMS_TO_TICKS(I2S_READ_TIMEOUT_MS));
        if (ret != ESP_OK) {
            gpio_set_level(DBG_REC_PIN, 0);
            ESP_LOGE(TAG, "i2s_channel_read failed: %s", esp_err_to_name(ret));
            return ret;
        }

        int n = bytes_read / sizeof(int32_t);
        for (int i = 0; i < n && total_samples < target_samples; i++) {
            int16_t pcm = (int16_t)(g_i2s_read_buffer[i] >> PCM_SHIFT_BITS);
            int32_t gained = (int32_t)pcm * PCM_GAIN;
            pcm_buffer[total_samples++] =
                (gained > INT16_MAX) ? INT16_MAX :
                (gained < INT16_MIN) ? INT16_MIN :
                (int16_t)gained;
        }
    }

    gpio_set_level(DBG_REC_PIN, 0);
    *samples_recorded = total_samples;

    double elapsed_s = (esp_timer_get_time() - t0) / 1000000.0;
    ESP_LOGI(TAG, "Capture done: %u samples, %.3f s",
             (unsigned)total_samples, elapsed_s);
    return ESP_OK;
}

static esp_err_t write_wav_after_capture(const char *path,
                                         const int16_t *pcm_buffer,
                                         size_t samples)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        ESP_LOGE(TAG, "Cannot open file: %s", path);
        return ESP_FAIL;
    }
    setvbuf(f, g_file_io_buffer, _IOFBF, sizeof(g_file_io_buffer));

    wav_header_t hdr = {0};
    fill_wav_header(&hdr, samples * sizeof(int16_t));

    gpio_set_level(DBG_SD_PIN, 1);
    size_t header_written = fwrite(&hdr, 1, sizeof(hdr), f);
    size_t samples_written = fwrite(pcm_buffer, sizeof(int16_t), samples, f);
    fflush(f);
    fclose(f);
    gpio_set_level(DBG_SD_PIN, 0);

    if (header_written != sizeof(hdr) || samples_written != samples) {
        ESP_LOGE(TAG, "Short write: header=%u/%u samples=%u/%u",
                 (unsigned)header_written, (unsigned)sizeof(hdr),
                 (unsigned)samples_written, (unsigned)samples);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "WAV written: %s | samples=%u | seconds=%.3f",
             path, (unsigned)samples, (double)samples / SAMPLE_RATE);
    return ESP_OK;
}

void app_main(void)
{
    ESP_LOGI(TAG, "Free internal heap: %u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT));
#ifdef CONFIG_SPIRAM
    ESP_LOGI(TAG, "Free PSRAM: %u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#endif

    dbg_pin_init();

    g_btn_pressed_sem = xSemaphoreCreateBinary();
    g_record_start_sem = xSemaphoreCreateBinary();
    if (!g_btn_pressed_sem || !g_record_start_sem) {
        ESP_LOGE(TAG, "Semaphore create failed");
        return;
    }

    btn_init();
    xTaskCreatePinnedToCore(button_task, "btn_task", 2048, NULL, 5, NULL, 0);

    if (sd_card_init() != ESP_OK) {
        ESP_LOGE(TAG, "SD Card initialization failed");
        return;
    }
    ESP_LOGI(TAG, "SD Card initialized");

    if (i2s_init() != ESP_OK) {
        ESP_LOGE(TAG, "I2S initialization failed");
        sd_card_deinit();
        return;
    }
    ESP_LOGI(TAG, "I2S initialized");

    const size_t target_samples = (size_t)SAMPLE_RATE * RECORD_TIME_SEC;
    const size_t pcm_bytes = target_samples * sizeof(int16_t);
    int16_t *pcm_buffer = (int16_t *)heap_caps_malloc(pcm_bytes, MALLOC_CAP_8BIT);
    if (!pcm_buffer) {
        ESP_LOGE(TAG, "Failed to allocate %u bytes for PCM buffer", (unsigned)pcm_bytes);
        i2s_deinit();
        sd_card_deinit();
        return;
    }
    ESP_LOGI(TAG, "Allocated PCM buffer: %u bytes", (unsigned)pcm_bytes);
    ESP_LOGI(TAG, "=== Ready. Press button to record %d sec into RAM, then write SD ===",
             RECORD_TIME_SEC);

    while (1) {
        xSemaphoreTake(g_record_start_sem, portMAX_DELAY);

        int file_idx = get_next_file_index();
        snprintf(g_wav_path, sizeof(g_wav_path), "/sdcard/%d.wav", file_idx);
        ESP_LOGI(TAG, ">>> Single-buffer recording #%d -> %s <<<", file_idx, g_wav_path);

        overflow_count = 0;
        size_t samples_recorded = 0;

        esp_err_t ret = record_to_ram(pcm_buffer, target_samples, &samples_recorded);
        if (ret == ESP_OK) {
            ret = write_wav_after_capture(g_wav_path, pcm_buffer, samples_recorded);
        }

        ESP_LOGI(TAG, "Finished: %s | result=%s | overflow=%u | samples=%u",
                 g_wav_path, esp_err_to_name(ret),
                 (unsigned)overflow_count,
                 (unsigned)samples_recorded);
        ESP_LOGI(TAG, "=== Press button to record again ===");
    }

    free(pcm_buffer);
    i2s_deinit();
    sd_card_deinit();
}
