/**
 * @file app_audio_infer.c
 * @brief Standard-I2S microphone capture, WAV archival, and float32 CNN inference.
 *
 * Runtime sequence:
 *   1. Initialize button, standard I2S RX, and SD card.
 *   2. Allocate persistent PCM, model-input, and DSP workspace buffers.
 *   3. Wait for a debounced button event.
 *   4. Capture exactly 32000 PCM16 samples into RAM.
 *   5. Save the exact inference PCM buffer as a WAV for desktop parity checks.
 *   6. Extract normalized log-Mel CMVN features and invoke TFLM.
 *
 * Concurrency model:
 *   - GPIO ISR only gives a semaphore and performs no blocking work.
 *   - button_task debounces input and notifies app_main.
 *   - app_main exclusively owns capture, storage, preprocessing, and inference.
 *
 * @warning I2S remains enabled while idle. record_to_ram() flushes stale DMA
 *          data before each capture. Moving SD writes into the capture loop can
 *          reintroduce storage-induced sampling jitter and must be avoided.
 */
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "driver/sdmmc_host.h"
#include "driver/i2s_std.h"

#include "logmel_features.h"
#include "memory_monitor.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "model_runner.h"
#include "preprocess_config.h"
#include "sdmmc_cmd.h"

/* Board wiring. Keep pin changes local to this hardware integration section. */
#define I2S_BCLK_IO             GPIO_NUM_26
#define I2S_WS_IO               GPIO_NUM_25
#define I2S_DIN_IO              GPIO_NUM_33

#define BTN_PIN                 GPIO_NUM_0
#define DBG_REC_PIN             GPIO_NUM_12
#define DBG_SD_PIN              GPIO_NUM_18

#define DMA_DESC_NUM            8
#define DMA_FRAME_NUM           512
#define I2S_READ_BYTES          8096
#define I2S_FLUSH_READS         4
#define I2S_READ_TIMEOUT_MS     1000

#define SD_MOUNT_POINT          "/sdcard"
#define WAV_BITS_PER_SAMPLE     16
#define WAV_CHANNELS            1
#define FILE_IO_BUFFER_SIZE     32768

/*
 * Keep recorder_single.c's microphone conversion path so firmware inference
 * sees the same 16-bit PCM scale as the WAV data used for training.
 */
#define PCM_SHIFT_BITS          16
#define PCM_GAIN                30

#define BTN_DEBOUNCE_MS         50
#define BTN_POST_RELEASE_MS     300

static const char *TAG = "AUDIO_INFER";

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

/* Process-lifetime driver handles and synchronization objects. */
static i2s_chan_handle_t rx_handle = NULL;
static sdmmc_card_t *sd_card = NULL;
static SemaphoreHandle_t g_btn_pressed_sem = NULL;
static SemaphoreHandle_t g_record_start_sem = NULL;

static volatile uint32_t overflow_count = 0;
/* Static internal-RAM buffers required by I2S and buffered FAT file output. */
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
    hdr->fmt_chunk_size = 16;
    hdr->audio_format = 1;  // PCM
    hdr->num_channels = WAV_CHANNELS;
    hdr->sample_rate = AUDIO_LOGMEL_SAMPLE_RATE_HZ;
    hdr->bits_per_sample = WAV_BITS_PER_SAMPLE;
    hdr->byte_rate = AUDIO_LOGMEL_SAMPLE_RATE_HZ * WAV_CHANNELS * (WAV_BITS_PER_SAMPLE / 8);
    hdr->block_align = WAV_CHANNELS * (WAV_BITS_PER_SAMPLE / 8);
    hdr->data_size = data_size;
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

static int get_next_wav_index(void)
{
    int idx = 1;
    char path[32];

    while (1) {
        snprintf(path, sizeof(path), SD_MOUNT_POINT "/%d.wav", idx);
        FILE *f = fopen(path, "rb");
        if (!f) {
            return idx;
        }
        fclose(f);
        idx++;
    }
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

    esp_err_t ret = esp_vfs_fat_sdmmc_mount(
        SD_MOUNT_POINT,
        &host,
        &slot_config,
        &mount_config,
        &sd_card);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s", esp_err_to_name(ret));
        return ret;
    }

    sdmmc_card_print_info(stdout, sd_card);
    return ESP_OK;
}

static void sd_card_deinit(void)
{
    if (sd_card) {
        esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, sd_card);
        sd_card = NULL;
    }
}

static void IRAM_ATTR btn_isr_handler(void *arg)
{
    /* ISR contract: signal only; debounce and GPIO polling run in button_task. */
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
    /* Convert noisy GPIO edges into one capture request per full press/release. */
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
        xSemaphoreGive(g_record_start_sem);
    }
}

static bool IRAM_ATTR i2s_rx_overflow_callback(i2s_chan_handle_t handle,
                                                i2s_event_data_t *event,
                                                void *user_ctx)
{
    /* Counter is diagnostic only; nonzero values indicate capture-path stress. */
    (void)handle;
    (void)event;
    (void)user_ctx;
    overflow_count++;
    return false;
}

/** @brief Configure mono 32-bit Philips I2S RX and enable continuous DMA. */
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
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_LOGMEL_SAMPLE_RATE_HZ),
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

/**
 * @brief Capture a fixed sample count and convert microphone words to PCM16.
 * @param[out] pcm_buffer Caller-owned buffer with target_samples capacity.
 * @param[in] target_samples Exact number of PCM16 samples to produce.
 * @param[out] samples_recorded Number of valid samples written on success.
 * @return ESP_OK on a complete capture; I2S read error otherwise.
 *
 * The shift/gain/saturation path is part of the model input contract because
 * training and desktop parity WAVs use this exact PCM scale.
 */
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

/**
 * @brief Persist the completed RAM capture as mono PCM16 WAV.
 * @note File I/O begins only after capture, so SD activity cannot disturb I2S.
 */
static esp_err_t write_wav_after_capture(const char *path,
                                         const int16_t *pcm_buffer,
                                         size_t samples)
{
    if (!path || !pcm_buffer || samples == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        ESP_LOGE(TAG, "Cannot open WAV file: %s", path);
        return ESP_FAIL;
    }

    /*
     * This writes the exact int16 PCM buffer used for inference.
     * Python-side comparison should read this WAV without additional gain/shift.
     */
    setvbuf(f, g_file_io_buffer, _IOFBF, sizeof(g_file_io_buffer));

    wav_header_t hdr = {0};
    fill_wav_header(&hdr, (uint32_t)(samples * sizeof(int16_t)));

    gpio_set_level(DBG_SD_PIN, 1);
    size_t header_written = fwrite(&hdr, 1, sizeof(hdr), f);
    size_t samples_written = fwrite(pcm_buffer, sizeof(int16_t), samples, f);
    fflush(f);
    fclose(f);
    gpio_set_level(DBG_SD_PIN, 0);

    if (header_written != sizeof(hdr) || samples_written != samples) {
        ESP_LOGE(TAG, "WAV short write: header=%u/%u samples=%u/%u",
                 (unsigned)header_written, (unsigned)sizeof(hdr),
                 (unsigned)samples_written, (unsigned)samples);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "WAV saved for Python check: %s | samples=%u | seconds=%.3f",
             path,
             (unsigned)samples,
             (double)samples / AUDIO_LOGMEL_SAMPLE_RATE_HZ);
    return ESP_OK;
}

/**
 * @brief Execute frontend and float32 TFLM inference for one capture.
 * @note The first call lazily allocates the persistent tensor arena.
 */
static esp_err_t run_audio_inference(const int16_t *pcm_buffer,
                                     size_t samples,
                                     float *model_input,
                                     void *feature_workspace,
                                     size_t feature_workspace_bytes)
{
    float probabilities[AUDIO_MODEL_CLASS_COUNT];
    int predicted_class = -1;

    memory_monitor_checkpoint("FEATURE_BEGIN");
    int64_t feature_start_us = esp_timer_get_time();
    int feat_ret = audio_logmel_extract_normalized_pcm16(
        pcm_buffer,
        samples,
        model_input,
        feature_workspace,
        feature_workspace_bytes,
        NULL);
    memory_monitor_checkpoint("FEATURE_END");
    if (feat_ret != AUDIO_LOGMEL_OK) {
        ESP_LOGE(TAG, "Feature extraction failed: %d", feat_ret);
        return ESP_FAIL;
    }
    int64_t feature_elapsed_us = esp_timer_get_time() - feature_start_us;

    memory_monitor_checkpoint("MODEL_PREDICT_BEGIN");
    int64_t inference_start_us = esp_timer_get_time();
    esp_err_t ret = model_runner_predict(
        model_input,
        probabilities,
        &predicted_class);
    memory_monitor_checkpoint("MODEL_PREDICT_END");
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Model prediction failed: %s", esp_err_to_name(ret));
        return ret;
    }
    int64_t inference_elapsed_us = esp_timer_get_time() - inference_start_us;

    ESP_LOGI(TAG, "Prediction: %s | background=%.4f drone=%.4f helicopter=%.4f",
             g_audio_class_names[predicted_class],
             probabilities[0],
             probabilities[1],
             probabilities[2]);
    ESP_LOGI(TAG, "Timing: logmel=%.2f ms tflite=%.2f ms",
             feature_elapsed_us / 1000.0,
             inference_elapsed_us / 1000.0);
    return ESP_OK;
}

void app_main(void)
{
    /* Log capacities before application-owned tasks/drivers allocate memory. */
    memory_monitor_log_capacities();
    memory_monitor_checkpoint("BOOT_ENTRY");

    dbg_pin_init();

    g_btn_pressed_sem = xSemaphoreCreateBinary();
    g_record_start_sem = xSemaphoreCreateBinary();
    if (!g_btn_pressed_sem || !g_record_start_sem) {
        ESP_LOGE(TAG, "Semaphore create failed");
        return;
    }

    btn_init();
    xTaskCreatePinnedToCore(button_task, "btn_task", 2048, NULL, 5, NULL, 0);
    memory_monitor_checkpoint("CONTROL_TASKS_READY");

    memory_monitor_checkpoint("I2S_INIT_BEGIN");
    esp_err_t ret = i2s_init();
    memory_monitor_checkpoint("I2S_INIT_END");
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2S initialization failed");
        return;
    }

    memory_monitor_checkpoint("SD_MOUNT_BEGIN");
    ret = sd_card_init();
    memory_monitor_checkpoint("SD_MOUNT_END");
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SD Card initialization failed");
        i2s_deinit();
        return;
    }

    /* These buffers persist to avoid per-capture allocation and fragmentation. */
    const size_t target_samples = AUDIO_LOGMEL_INPUT_SAMPLES;
    const size_t pcm_bytes = target_samples * sizeof(int16_t);
    memory_monitor_checkpoint("PIPELINE_ALLOC_BEGIN");
    int16_t *pcm_buffer = (int16_t *)heap_caps_malloc_prefer(
        pcm_bytes,
        2,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
        MALLOC_CAP_8BIT);
    memory_monitor_allocation("pcm_buffer", pcm_buffer, pcm_bytes);
    if (!pcm_buffer) {
        ESP_LOGE(TAG, "Failed to allocate %u bytes for PCM buffer", (unsigned)pcm_bytes);
        i2s_deinit();
        sd_card_deinit();
        return;
    }

    const size_t model_input_bytes =
        AUDIO_LOGMEL_MODEL_INPUT_VALUES * sizeof(float);
    float *model_input = (float *)heap_caps_malloc_prefer(
        model_input_bytes,
        2,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
        MALLOC_CAP_8BIT);
    memory_monitor_allocation("model_input", model_input, model_input_bytes);
    const size_t feature_workspace_bytes = audio_logmel_workspace_size();
    void *feature_workspace = heap_caps_malloc_prefer(
        feature_workspace_bytes,
        2,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
        MALLOC_CAP_8BIT);
    memory_monitor_allocation("feature_workspace",
                              feature_workspace,
                              feature_workspace_bytes);
    memory_monitor_checkpoint("PIPELINE_ALLOC_END");
    if (!model_input || !feature_workspace) {
        ESP_LOGE(TAG, "Log-mel buffers allocation failed: input=%u workspace=%u",
                 (unsigned)model_input_bytes,
                 (unsigned)feature_workspace_bytes);
        free(model_input);
        free(feature_workspace);
        free(pcm_buffer);
        i2s_deinit();
        sd_card_deinit();
        return;
    }

    ESP_LOGI(TAG, "Ready. Press button to record %d sec, save WAV, and infer.",
             AUDIO_LOGMEL_RECORD_SECONDS);

    /* app_main serializes each complete capture/save/inference transaction. */
    while (1) {
        xSemaphoreTake(g_record_start_sem, portMAX_DELAY);
        memory_monitor_checkpoint("CYCLE_BEGIN");

        int file_idx = get_next_wav_index();
        snprintf(g_wav_path, sizeof(g_wav_path), SD_MOUNT_POINT "/%d.wav", file_idx);
        ESP_LOGI(TAG, "Recording target WAV: %s", g_wav_path);

        overflow_count = 0;
        size_t samples_recorded = 0;
        memory_monitor_checkpoint("CAPTURE_BEGIN");
        ret = record_to_ram(pcm_buffer, target_samples, &samples_recorded);
        memory_monitor_checkpoint("CAPTURE_END");
        if (ret == ESP_OK) {
            memory_monitor_checkpoint("WAV_WRITE_BEGIN");
            esp_err_t wav_ret = write_wav_after_capture(
                g_wav_path,
                pcm_buffer,
                samples_recorded);
            if (wav_ret != ESP_OK) {
                ESP_LOGE(TAG, "WAV save failed before inference: %s",
                          esp_err_to_name(wav_ret));
            }
            memory_monitor_checkpoint("WAV_WRITE_END");

            ret = run_audio_inference(
                pcm_buffer,
                samples_recorded,
                model_input,
                feature_workspace,
                feature_workspace_bytes);
        }

        ESP_LOGI(TAG, "Done: wav=%s result=%s overflow=%u samples=%u",
                 g_wav_path,
                 esp_err_to_name(ret),
                  (unsigned)overflow_count,
                  (unsigned)samples_recorded);
        memory_monitor_checkpoint("CYCLE_END");
        ESP_LOGI(TAG, "Press button to run again.");
    }

    free(feature_workspace);
    free(model_input);
    free(pcm_buffer);
    i2s_deinit();
    sd_card_deinit();
}
