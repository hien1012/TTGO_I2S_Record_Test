#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/unistd.h>
#include <errno.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "driver/sdspi_host.h"
#include "driver/i2s_pdm.h"

#include "sdmmc_cmd.h"

static const char *TAG = "I2S_REC";

// =========================
// 硬體腳位配置 (針對 TTGO T1 ESP32)
// =========================
// 1. SD 卡 SPI 腳位 (使用標準 VSPI 腳位)
#define PIN_SD_MOSI     15
#define PIN_SD_MISO     2
#define PIN_SD_SCLK     14
#define PIN_SD_CS       13

// 2. ICS43434 I2S 麥克風腳位 (選擇避開系統保留與 SPI 的安全腳位)
#define PIN_MIC_BCLK    26  // 接至麥克風 BCLK
#define PIN_MIC_LRCL    25  // 接至麥克風 LRCL (WS)
#define PIN_MIC_DOUT    33  // 接至麥克風 DOUT (ESP32 的 DIN)

// 3. LED 腳位 (TTGO T1 通常沒有獨立的 User LED，這裡指定一個腳位，若無連接可忽略)
#define PIN_LED         22

// =========================
// 使用者設定
// =========================
#define SAMPLE_RATE_HZ          16000
#define RECORD_TIME_SEC         3
#define NUM_CHANNELS            1
#define BITS_PER_SAMPLE         16

#define I2S_READ_SAMPLES        512
#define I2S_READ_BYTES          (I2S_READ_SAMPLES * sizeof(int16_t))

#define SD_MOUNT_POINT          "/sdcard"

// =========================
// 全域變數
// =========================
static sdmmc_card_t *g_card = NULL;
static i2s_chan_handle_t g_rx_handle = NULL;
static int16_t g_i2s_buf[I2S_READ_SAMPLES];

// =========================
// WAV 標頭結構
// =========================
typedef struct __attribute__((packed)) {
    char riff[4];
    uint32_t overall_size;
    char wave[4];
    char fmt_chunk_marker[4];
    uint32_t length_of_fmt;
    uint16_t format_type;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byterate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data_chunk_header[4];
    uint32_t data_size;
} wav_header_t;

static wav_header_t make_wav_header(uint32_t data_size)
{
    wav_header_t h;
    memcpy(h.riff, "RIFF", 4);
    memcpy(h.wave, "WAVE", 4);
    memcpy(h.fmt_chunk_marker, "fmt ", 4);
    memcpy(h.data_chunk_header, "data", 4);

    h.overall_size = data_size + sizeof(wav_header_t) - 8;
    h.length_of_fmt = 16;
    h.format_type = 1; // PCM
    h.channels = NUM_CHANNELS;
    h.sample_rate = SAMPLE_RATE_HZ;
    h.bits_per_sample = BITS_PER_SAMPLE;
    h.byterate = SAMPLE_RATE_HZ * NUM_CHANNELS * (BITS_PER_SAMPLE / 8);
    h.block_align = NUM_CHANNELS * (BITS_PER_SAMPLE / 8);
    h.data_size = data_size;

    return h;
}

// =========================
// LED 功能
// =========================
static void led_init(void)
{
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << PIN_LED,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io);
    gpio_set_level(PIN_LED, 0);
}

static void blink_led(int times, int delay_ms)
{
    for (int i = 0; i < times; i++) {
        gpio_set_level(PIN_LED, 1);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
        gpio_set_level(PIN_LED, 0);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }
}

// =========================
// SD 卡初始化 (SPI 模式)
// =========================
static esp_err_t init_sdcard(void)
{
    ESP_LOGI(TAG, "Initializing SD card (SPI mode)...");

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = PIN_SD_MOSI,
        .miso_io_num = PIN_SD_MISO,
        .sclk_io_num = PIN_SD_SCLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };

    // TTGO T1 使用 ESP32，建議使用 SPI3_HOST (VSPI) 或 SPI2_HOST (HSPI)
    esp_err_t ret = spi_bus_initialize(SPI2_HOST, &bus_cfg, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "spi_bus_initialize failed: %s", esp_err_to_name(ret));
        return ret;
    }

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = SPI2_HOST;

    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = PIN_SD_CS;
    slot_config.host_id = SPI2_HOST;

    esp_vfs_fat_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024,
    };

    ret = esp_vfs_fat_sdspi_mount(SD_MOUNT_POINT, &host, &slot_config, &mount_config, &g_card);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_vfs_fat_sdspi_mount failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "SD card mounted at %s", SD_MOUNT_POINT);
    sdmmc_card_print_info(stdout, g_card);
    return ESP_OK;
}

static void deinit_sdcard(void)
{
    if (g_card) {
        esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, g_card);
        g_card = NULL;
        ESP_LOGI(TAG, "SD card unmounted");
    }
    spi_bus_free(SPI2_HOST);
}

// =========================
// I2S 標準麥克風初始化 (ICS43434)
// =========================
static esp_err_t init_microphone(void)
{
    ESP_LOGI(TAG, "Initializing PDM microphone...");

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = 240;

    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &g_rx_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel failed: %s", esp_err_to_name(ret));
        return ret;
    }

    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            // 沿用你原本的腳位名稱，不改 define 名稱
            .clk = PIN_MIC_BCLK,
            .din = PIN_MIC_DOUT,
            .invert_flags = {
                .clk_inv = false,
            },
        },
    };

    ret = i2s_channel_init_pdm_rx_mode(g_rx_handle, &pdm_rx_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_init_pdm_rx_mode failed: %s", esp_err_to_name(ret));
        i2s_del_channel(g_rx_handle);
        g_rx_handle = NULL;
        return ret;
    }

    ret = i2s_channel_enable(g_rx_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_enable failed: %s", esp_err_to_name(ret));
        i2s_del_channel(g_rx_handle);
        g_rx_handle = NULL;
        return ret;
    }

    ESP_LOGI(TAG, "PDM microphone ready. CLK=%d DATA=%d", PIN_MIC_BCLK, PIN_MIC_DOUT);
    vTaskDelay(pdMS_TO_TICKS(500));
    return ESP_OK;
}

static void deinit_microphone(void)
{
    if (g_rx_handle) {
        i2s_channel_disable(g_rx_handle);
        i2s_del_channel(g_rx_handle);
        g_rx_handle = NULL;
        ESP_LOGI(TAG, "Microphone deinitialized");
    }
}

// =========================
// 檔案命名
// =========================
static int find_next_wav_index(const char *base_path)
{
    DIR *dir = opendir(base_path);
    if (!dir) return 1;

    int max_idx = 0;
    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        int idx = 0;
        if (sscanf(entry->d_name, "rec%d.wav", &idx) == 1) {
            if (idx > max_idx) max_idx = idx;
        }
    }
    closedir(dir);
    return max_idx + 1;
}

// =========================
// 錄音並寫入 WAV 檔案
// =========================
static esp_err_t record_wav(const char *filepath, uint32_t seconds)
{
    struct stat st = {0};
    if (stat(SD_MOUNT_POINT, &st) == -1) {
        mkdir(SD_MOUNT_POINT, 0777);
    }
    
    FILE *f = fopen(filepath, "wb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open file: %s", filepath);
        return ESP_FAIL;
    }

    wav_header_t header = make_wav_header(0);
    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        ESP_LOGE(TAG, "Failed to write WAV header");
        fclose(f);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Start recording: %s for %lu sec", filepath, seconds);

    size_t total_bytes_written = 0;
    int64_t start_us = esp_timer_get_time();

    while (1) {
        int64_t now_us = esp_timer_get_time();
        if ((now_us - start_us) >= ((int64_t)seconds * 1000000LL)) break;

        size_t bytes_read = 0;
        esp_err_t ret = i2s_channel_read(g_rx_handle, g_i2s_buf, I2S_READ_BYTES, &bytes_read, pdMS_TO_TICKS(500));

        if (ret == ESP_OK) {
            if (bytes_read > 0) {
                size_t written = fwrite(g_i2s_buf, 1, bytes_read, f);
                if (written != bytes_read) {
                    ESP_LOGE(TAG, "File write failed");
                    fclose(f);
                    return ESP_FAIL;
                }
                total_bytes_written += written;
            }
        } else if (ret == ESP_ERR_TIMEOUT) {
            continue;
        } else {
            ESP_LOGE(TAG, "i2s_channel_read failed: %s", esp_err_to_name(ret));
            fclose(f);
            return ret;
        }
    }

    // 更新正確的檔案大小標頭
    header = make_wav_header((uint32_t)total_bytes_written);
    fseek(f, 0, SEEK_SET);
    fwrite(&header, sizeof(header), 1, f);
    fclose(f);

    ESP_LOGI(TAG, "Recording done. WAV bytes: %u", (unsigned)total_bytes_written);
    return ESP_OK;
}

// =========================
// 主程式
// =========================
void app_main(void)
{
    ESP_LOGI(TAG, "=== I2S Standard recorder start ===");

    led_init();
    blink_led(2, 100);

    esp_err_t ret;

    ret = init_sdcard();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SD init failed");
        while (1) { blink_led(1, 500); vTaskDelay(pdMS_TO_TICKS(1000)); }
    }

    ret = init_microphone();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Microphone init failed");
        deinit_sdcard();
        while (1) { blink_led(2, 300); vTaskDelay(pdMS_TO_TICKS(1000)); }
    }

    ESP_LOGI(TAG, "Waiting for microphone to stabilize...");
    vTaskDelay(pdMS_TO_TICKS(1000));

    int index = find_next_wav_index(SD_MOUNT_POINT);
    char filepath[128];
    snprintf(filepath, sizeof(filepath), "%s/rec%03d.wav", SD_MOUNT_POINT, index);

    ret = record_wav(filepath, RECORD_TIME_SEC);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Saved file: %s", filepath);
        blink_led(3, 120);
    } else {
        ESP_LOGE(TAG, "Recording failed");
        blink_led(5, 120);
    }

    deinit_microphone();
    deinit_sdcard();

    ESP_LOGI(TAG, "All done");

    while (1) {
        gpio_set_level(PIN_LED, 1);
        vTaskDelay(pdMS_TO_TICKS(500));
        gpio_set_level(PIN_LED, 0);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}