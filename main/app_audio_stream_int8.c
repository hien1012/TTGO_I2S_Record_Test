/**
 * @file app_audio_stream_int8.c
 * @brief Continuous I2S/PDM capture with full-INT8 inference on the other core.
 *
 * Microphone: standard I2S by default; build with -DAUDIO_MIC=pdm for PDM
 * (CMake defines STREAM_MIC_PDM=1). Both use the same PCM conversion below.
 *
 * Design ported from DID_NODE/infer_node/main/mic_infer.c (ESP32-S3), adapted
 * to this ESP32 board: no button, no Wi-Fi. Recording starts at boot and runs
 * forever in 2 s windows; while window N is being captured, window N-1 is
 * inferred.
 *
 *   mic_capture  (Core 0, prio 7): I2S DMA holds only ~256 ms, so highest prio.
 *       I2S read -> PCM x30 (saturated) -> current window; when full, hand it
 *       to logmel_task and continue in the next free window.
 *   logmel_task  (Core 0, prio 4): log-Mel CMVN + int8 quantization into a
 *       feature buffer, handed to cnn_task. Below capture so I2S reads are
 *       never delayed.
 *   cnn_task     (Core 1, prio 3): INT8 CNN on that feature buffer, log the
 *       result. Windows to be saved are handed to audio_writer.
 *   audio_writer (Core 1, prio 2): write WAV to SD, then return the window.
 *       SD writes can take hundreds of ms; they never count against the 2 s
 *       budget because cnn_task preempts this task.
 *
 * Four 64 KB windows live in PSRAM (capture 1 + logmel 1 + CNN 1 + SD write 1);
 * a window is held until the CNN result decides whether it is saved. Indices
 * cycle s_free_windows -> s_full_windows -> s_feature_jobs -> (s_save_jobs)
 * -> s_free_windows. Two int8 feature buffers let logmel of window N overlap
 * the CNN of window N-1. If all windows are busy, new audio is dropped and
 * counted; I2S is never blocked.
 *
 * PCM contract (identical to app_audio_infer_int8.c / app_audio_infer_pdm_int8.c
 * and their training WAVs): both mics read 32-bit slots, then
 * (int16)(raw32 >> 16) * 30, saturated. Saved WAVs are the exact model input.
 * Note: DID_NODE's PDM path reads 16-bit slots without the shift; this project
 * deliberately keeps its own recorder's 32-bit convention.
 */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "driver/sdmmc_host.h"
#if STREAM_MIC_PDM
#include "driver/i2s_pdm.h"
#else
#include "driver/i2s_std.h"
#endif
#include "esp_attr.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"

#include "logmel_features_int8.h"
#include "memory_monitor.h"
#include "model_runner_int8.h"
#include "preprocess_config_int8.h"

/* ---------------------------------------------------------------------------
 * Configuration
 * ------------------------------------------------------------------------- */

/* Which windows are written to SD as WAV. */
#define SAVE_NONE               0
#define SAVE_LOW_CONFIDENCE     1   /* predicted-class probability < threshold */
#define SAVE_ALL                2
#define STREAM_SAVE_MODE        SAVE_LOW_CONFIDENCE
#define LOW_CONF_THRESHOLD      0.5f

/* Board wiring (same as app_audio_infer_int8.c / app_audio_infer_pdm_int8.c). */
#if STREAM_MIC_PDM
#define MIC_NAME                "PDM"
#define PDM_CLK_IO              GPIO_NUM_4
#define PDM_DIN_IO              GPIO_NUM_5
#else
#define MIC_NAME                "I2S"
#define I2S_BCLK_IO             GPIO_NUM_26
#define I2S_WS_IO               GPIO_NUM_25
#define I2S_DIN_IO              GPIO_NUM_33
#endif
#define DBG_CNN_PIN             GPIO_NUM_12     /* high while the CNN runs */
#define DBG_SD_PIN              GPIO_NUM_18     /* high while a WAV is written */

#define MIC_SAMPLE_RATE         AUDIO_LOGMEL_SAMPLE_RATE_HZ
#define WINDOW_SAMPLES          AUDIO_LOGMEL_INPUT_SAMPLES
#define WINDOW_BYTES            (WINDOW_SAMPLES * sizeof(int16_t))
#define WINDOW_COUNT            4U
#define FEATURE_COUNT           2U      /* logmel fills one while the CNN reads the other */

/* I2S DMA: 8 x 512 frames = 256 ms of buffering. */
#define DMA_DESC_NUM            8
#define DMA_FRAME_NUM           512
#define RAW_BYTES_PER_SAMPLE    4U      /* 32-bit slot, PCM in the upper 16 bits */
#define READ_SAMPLES            1024U   /* 64 ms per read */
#define READ_TIMEOUT_MS         200

/* Part of the model's input-data contract; never change on its own. */
#define PCM_SHIFT_BITS          16
#define PCM_GAIN                30
/* Microphone wake-up (INMP441 needs a few hundred ms); this audio is discarded. */
#define MIC_SETTLE_US           500000LL

#define SD_MOUNT_POINT          "/sdcard"
#define SD_LOG_PATH             SD_MOUNT_POINT "/STREAM.CSV"
#define WAV_HEADER_BYTES        44U
#define WRITE_BUFFER_BYTES      (8U * 1024U)    /* internal RAM; SDMMC DMA cannot read PSRAM */
#define STATS_PERIOD_WINDOWS    30U             /* summary every 30 windows = 1 minute */

#define CAPTURE_TASK_STACK_BYTES 4096U
#define LOGMEL_TASK_STACK_BYTES  4096U
#define CNN_TASK_STACK_BYTES     8192U
#define WRITER_TASK_STACK_BYTES  4096U
#define CAPTURE_TASK_PRIORITY    7U
#define LOGMEL_TASK_PRIORITY     4U
#define CNN_TASK_PRIORITY        3U
#define WRITER_TASK_PRIORITY     2U
#if CONFIG_FREERTOS_UNICORE
#define CAPTURE_TASK_CORE        0
#define LOGMEL_TASK_CORE         0
#define CNN_TASK_CORE            0
#define WRITER_TASK_CORE         0
#else
#define CAPTURE_TASK_CORE        0
#define LOGMEL_TASK_CORE         0      /* below capture on the same core */
#define CNN_TASK_CORE            1
#define WRITER_TASK_CORE         1      /* below cnn_task on the same core */
#endif

_Static_assert((WRITE_BUFFER_BYTES % 512U) == 0U, "WAV writes must contain complete SD sectors");

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

_Static_assert(sizeof(wav_header_t) == WAV_HEADER_BYTES, "WAV header is 44 bytes");

/* One captured window handed from mic_capture to logmel_task. */
typedef struct {
    uint8_t buffer;
    uint32_t window;
    int64_t start_us;
    int64_t end_us;
    uint32_t overflows;
} full_window_t;

/* One quantized feature buffer handed from logmel_task to cnn_task. */
typedef struct {
    full_window_t win;          /* PCM window is still held (needed if it is saved) */
    uint8_t feature;
    uint32_t logmel_queue_us;   /* window full -> logmel start */
    uint32_t feature_us;
    int64_t feature_done_us;
} feature_job_t;

/* One window handed from cnn_task to audio_writer; the writer returns it. */
typedef struct {
    uint8_t buffer;
    uint32_t window;
    int pred;
    float probs[AUDIO_MODEL_CLASS_COUNT];
} save_job_t;

typedef struct {
    uint32_t windows;
    uint32_t saved;
    uint32_t save_failed;
    uint32_t feature_us_max;
    uint32_t invoke_us_max;
    uint64_t feature_us_sum;
    uint64_t invoke_us_sum;
} stream_stats_t;

static const char *TAG = "AUDIO_STREAM";

static i2s_chan_handle_t s_rx_handle = NULL;
static sdmmc_card_t *s_sd_card = NULL;
static bool s_sd_ready = false;
static uint32_t s_next_file_index = 1;
static volatile uint32_t s_overflow_count = 0;
static volatile uint32_t s_dropped_windows = 0;
static stream_stats_t s_stats;

static int16_t *s_windows[WINDOW_COUNT];        /* PSRAM */
static int8_t *s_features[FEATURE_COUNT];       /* PSRAM, quantized model inputs */
static void *s_feature_workspace = NULL;        /* PSRAM, logmel_task only */
static size_t s_feature_workspace_bytes = 0;

static StaticQueue_t s_free_queue_control;
static StaticQueue_t s_full_queue_control;
static StaticQueue_t s_free_feature_queue_control;
static StaticQueue_t s_feature_queue_control;
static StaticQueue_t s_save_queue_control;
static uint8_t s_free_queue_storage[WINDOW_COUNT * sizeof(uint8_t)];
static uint8_t s_full_queue_storage[WINDOW_COUNT * sizeof(full_window_t)];
static uint8_t s_free_feature_queue_storage[FEATURE_COUNT * sizeof(uint8_t)];
static uint8_t s_feature_queue_storage[FEATURE_COUNT * sizeof(feature_job_t)];
static uint8_t s_save_queue_storage[WINDOW_COUNT * sizeof(save_job_t)];
static QueueHandle_t s_free_windows = NULL;
static QueueHandle_t s_full_windows = NULL;     /* depth = window count, never full */
static QueueHandle_t s_free_features = NULL;
static QueueHandle_t s_feature_jobs = NULL;     /* depth = feature count, never full */
static QueueHandle_t s_save_jobs = NULL;        /* depth = window count, never full */

static WORD_ALIGNED_ATTR uint8_t s_read_buffer[READ_SAMPLES * RAW_BYTES_PER_SAMPLE];
static WORD_ALIGNED_ATTR uint8_t s_write_buffer[WRITE_BUFFER_BYTES];   /* audio_writer only */

static StaticTask_t s_capture_tcb;
static StaticTask_t s_logmel_tcb;
static StaticTask_t s_cnn_tcb;
static StaticTask_t s_writer_tcb;
static StackType_t s_capture_stack[CAPTURE_TASK_STACK_BYTES];
static StackType_t s_logmel_stack[LOGMEL_TASK_STACK_BYTES];
static StackType_t s_cnn_stack[CNN_TASK_STACK_BYTES];
static StackType_t s_writer_stack[WRITER_TASK_STACK_BYTES];
static TaskHandle_t s_capture_task = NULL;
static TaskHandle_t s_logmel_task = NULL;
static TaskHandle_t s_cnn_task = NULL;
static TaskHandle_t s_writer_task = NULL;

/* ---------------------------------------------------------------------------
 * Drivers: debug pins, SD card, I2S
 * ------------------------------------------------------------------------- */

static void dbg_pin_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << DBG_CNN_PIN) | (1ULL << DBG_SD_PIN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    gpio_set_level(DBG_CNN_PIN, 0);
    gpio_set_level(DBG_SD_PIN, 0);
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

    ESP_RETURN_ON_ERROR(esp_vfs_fat_sdmmc_mount(SD_MOUNT_POINT, &host, &slot_config,
                                                &mount_config, &s_sd_card),
                        TAG, "SD mount failed");
    sdmmc_card_print_info(stdout, s_sd_card);
    return ESP_OK;
}

/* FATFS is built without long file names, so names stay 8.3: S0000001.WAV. */
static void wav_path_for_index(uint32_t index, char *path, size_t path_size)
{
    snprintf(path, path_size, SD_MOUNT_POINT "/S%07" PRIu32 ".WAV", index);
}

/* Scan once at boot so a reboot continues numbering instead of overwriting. */
static uint32_t find_next_file_index(void)
{
    char path[32];
    struct stat st;
    uint32_t index = 1;
    for (;;) {
        wav_path_for_index(index, path, sizeof(path));
        if (stat(path, &st) != 0) {
            return index;
        }
        ++index;
    }
}

static bool IRAM_ATTR i2s_rx_overflow_callback(i2s_chan_handle_t handle,
                                               i2s_event_data_t *event,
                                               void *user_ctx)
{
    (void)handle; (void)event; (void)user_ctx;
    s_overflow_count++;
    return false;
}

/*
 * Mono receive with 32-bit slots:
 *   I2S: Philips standard, identical to app_audio_infer_int8.c.
 *   PDM: PDM-to-PCM RX, identical to app_audio_infer_pdm_int8.c.
 */
static esp_err_t i2s_init(void)
{
    i2s_chan_config_t chan_cfg = {
        .id            = I2S_NUM_0,
        .role          = I2S_ROLE_MASTER,
        .dma_desc_num  = DMA_DESC_NUM,
        .dma_frame_num = DMA_FRAME_NUM,
        .auto_clear    = true,
    };
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, NULL, &s_rx_handle),
                        TAG, "i2s_new_channel failed");

#if STREAM_MIC_PDM
    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg  = I2S_PDM_RX_CLK_DEFAULT_CONFIG(MIC_SAMPLE_RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                   I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = PDM_CLK_IO,
            .din = PDM_DIN_IO,
            .invert_flags = {
                .clk_inv = false,
            },
        },
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_pdm_rx_mode(s_rx_handle, &pdm_rx_cfg),
                        TAG, "i2s_channel_init_pdm_rx_mode failed");
#else
    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(MIC_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_MONO),
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
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    std_cfg.slot_cfg.ws_pol    = false;
    std_cfg.slot_cfg.bit_shift = true;
    std_cfg.slot_cfg.msb_right = false;
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_rx_handle, &std_cfg),
                        TAG, "i2s_channel_init_std_mode failed");
#endif

    const i2s_event_callbacks_t cbs = {
        .on_recv       = NULL,
        .on_recv_q_ovf = i2s_rx_overflow_callback,
        .on_sent       = NULL,
        .on_send_q_ovf = NULL,
    };
    ESP_RETURN_ON_ERROR(i2s_channel_register_event_callback(s_rx_handle, &cbs, NULL),
                        TAG, "register cb failed");
    return ESP_OK;
}

static inline int16_t raw_to_pcm(size_t index)
{
    const int32_t *raw = (const int32_t *)s_read_buffer;
    const int32_t gained = (int32_t)(int16_t)(raw[index] >> PCM_SHIFT_BITS) * PCM_GAIN;
    return gained > INT16_MAX ? INT16_MAX : gained < INT16_MIN ? INT16_MIN : (int16_t)gained;
}

/* ---------------------------------------------------------------------------
 * mic_capture (Core 0): read I2S continuously, hand each full window to logmel_task
 * ------------------------------------------------------------------------- */

static void mic_capture(void *arg)
{
    (void)arg;
    size_t bytes_read = 0;

    const int64_t settle_deadline_us = esp_timer_get_time() + MIC_SETTLE_US;
    while (esp_timer_get_time() < settle_deadline_us) {
        (void)i2s_channel_read(s_rx_handle, s_read_buffer, sizeof(s_read_buffer),
                               &bytes_read, READ_TIMEOUT_MS);
    }
    ESP_LOGI(TAG, "Capture running on core %d", xPortGetCoreID());

    int current = -1;           /* window being filled; -1 = all busy, audio is dropped */
    size_t fill = 0;
    uint32_t window_index = 0;
    int64_t window_start_us = 0;
    uint32_t window_start_overflows = 0;

    while (true) {
        const esp_err_t err = i2s_channel_read(s_rx_handle, s_read_buffer, sizeof(s_read_buffer),
                                               &bytes_read, READ_TIMEOUT_MS);
        const int64_t read_done_us = esp_timer_get_time();
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "i2s_channel_read failed: %s", esp_err_to_name(err));
            continue;
        }
        const size_t count = bytes_read / RAW_BYTES_PER_SAMPLE;
        for (size_t i = 0; i < count; ++i) {
            if (current < 0) {
                uint8_t next = 0;
                if (xQueueReceive(s_free_windows, &next, 0) != pdTRUE) {
                    break;      /* still no free window: drop the rest of this read */
                }
                current = next;
                fill = 0;
            }
            if (fill == 0U) {
                /* read returns just after its last sample; back-date the window start */
                window_start_us = read_done_us -
                                  (int64_t)(count - i) * 1000000LL / MIC_SAMPLE_RATE;
                window_start_overflows = s_overflow_count;
            }
            s_windows[current][fill++] = raw_to_pcm(i);
            if (fill == WINDOW_SAMPLES) {
                const full_window_t full = {
                    .buffer = (uint8_t)current,
                    .window = window_index++,
                    .start_us = window_start_us,
                    .end_us = read_done_us -
                              (int64_t)(count - i - 1U) * 1000000LL / MIC_SAMPLE_RATE,
                    .overflows = s_overflow_count - window_start_overflows,
                };
                (void)xQueueSend(s_full_windows, &full, 0);     /* depth = window count */
                current = -1;
                fill = 0;
                uint8_t next = 0;
                if (xQueueReceive(s_free_windows, &next, 0) == pdTRUE) {
                    current = next;
                } else {
                    ++s_dropped_windows;
                    ESP_LOGW(TAG, "All %u windows busy (logmel/CNN or SD too slow), dropped=%u",
                             (unsigned)WINDOW_COUNT, (unsigned)s_dropped_windows);
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * logmel_task (Core 0) -> cnn_task (Core 1) on the previous window
 * ------------------------------------------------------------------------- */

static bool should_save(float confidence)
{
#if STREAM_SAVE_MODE == SAVE_ALL
    (void)confidence;
    return true;
#elif STREAM_SAVE_MODE == SAVE_LOW_CONFIDENCE
    return confidence < LOW_CONF_THRESHOLD;
#else
    (void)confidence;
    return false;
#endif
}

static void log_stats(void)
{
    const uint32_t n = s_stats.windows;
    ESP_LOGI(TAG,
             "STATS windows=%u dropped=%u overflows=%u saved=%u save_failed=%u "
             "logmel avg/max=%u/%u ms cnn avg/max=%u/%u ms",
             (unsigned)n, (unsigned)s_dropped_windows, (unsigned)s_overflow_count,
             (unsigned)s_stats.saved, (unsigned)s_stats.save_failed,
             (unsigned)(s_stats.feature_us_sum / n / 1000U),
             (unsigned)(s_stats.feature_us_max / 1000U),
             (unsigned)(s_stats.invoke_us_sum / n / 1000U),
             (unsigned)(s_stats.invoke_us_max / 1000U));
    ESP_LOGI(TAG, "STATS stack_free bytes: capture=%u logmel=%u cnn=%u writer=%u; "
             "heap min_free internal=%u psram=%u",
             (unsigned)uxTaskGetStackHighWaterMark(s_capture_task),
             (unsigned)uxTaskGetStackHighWaterMark(s_logmel_task),
             (unsigned)uxTaskGetStackHighWaterMark(s_cnn_task),
             (unsigned)uxTaskGetStackHighWaterMark(s_writer_task),
             (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
}

static void release_window(uint8_t buffer)
{
    (void)xQueueSend(s_free_windows, &buffer, 0);
}

static void release_feature(uint8_t feature)
{
    (void)xQueueSend(s_free_features, &feature, 0);
}

/* Core 0: log-Mel CMVN + int8 quantization of the newest full window. */
static void logmel_task(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "Log-mel running on core %d", xPortGetCoreID());
    full_window_t full;
    while (true) {
        (void)xQueueReceive(s_full_windows, &full, portMAX_DELAY);
        /* Two feature buffers: if the CNN still holds both, wait for it here.
         * Further full windows keep queuing in s_full_windows meanwhile. */
        uint8_t feature = 0;
        (void)xQueueReceive(s_free_features, &feature, portMAX_DELAY);

        const int64_t started_us = esp_timer_get_time();
        const int err = audio_logmel_extract_quantized_pcm16(
            s_windows[full.buffer], WINDOW_SAMPLES, s_features[feature],
            s_feature_workspace, s_feature_workspace_bytes, NULL);
        const int64_t done_us = esp_timer_get_time();
        if (err != AUDIO_LOGMEL_OK) {
            ESP_LOGE(TAG, "w%u log-mel failed: %d", (unsigned)full.window, err);
            release_feature(feature);
            release_window(full.buffer);
            continue;
        }

        const feature_job_t job = {
            .win = full,
            .feature = feature,
            .logmel_queue_us = (uint32_t)(started_us - full.end_us),
            .feature_us = (uint32_t)(done_us - started_us),
            .feature_done_us = done_us,
        };
        (void)xQueueSend(s_feature_jobs, &job, 0);
    }
}

/* Core 1: INT8 CNN on a finished feature buffer; decides whether the window is saved. */
static void cnn_task(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "CNN running on core %d", xPortGetCoreID());
    feature_job_t job;
    while (true) {
        (void)xQueueReceive(s_feature_jobs, &job, portMAX_DELAY);
        const full_window_t *full = &job.win;
        gpio_set_level(DBG_CNN_PIN, 1);
        const int64_t started_us = esp_timer_get_time();

        float probs[AUDIO_MODEL_CLASS_COUNT] = {0};
        int pred = -1;
        const esp_err_t model_err = model_runner_int8_predict(s_features[job.feature], probs, &pred);
        const int64_t invoke_done_us = esp_timer_get_time();
        gpio_set_level(DBG_CNN_PIN, 0);
        release_feature(job.feature);   /* logmel_task can reuse it right away */

        if (model_err != ESP_OK || pred < 0) {
            ESP_LOGE(TAG, "w%u CNN failed: %s", (unsigned)full->window, esp_err_to_name(model_err));
            release_window(full->buffer);
            continue;
        }

        const uint32_t feature_us = job.feature_us;
        const uint32_t invoke_us = (uint32_t)(invoke_done_us - started_us);
        const uint32_t cnn_queue_us = (uint32_t)(started_us - job.feature_done_us);
        /* latency = window fully captured -> CNN result */
        const uint32_t latency_us = (uint32_t)(invoke_done_us - full->end_us);
        const bool save = s_sd_ready && should_save(probs[pred]);
        ESP_LOGI(TAG, "w%u %s %.3f [%.3f %.3f %.3f] logmel=%u ms cnn=%u ms "
                 "wait logmel/cnn=%u/%u ms latency=%u ms ovf=%u%s",
                 (unsigned)full->window, g_audio_class_names[pred], (double)probs[pred],
                 (double)probs[0], (double)probs[1], (double)probs[2],
                 (unsigned)(feature_us / 1000U), (unsigned)(invoke_us / 1000U),
                 (unsigned)(job.logmel_queue_us / 1000U), (unsigned)(cnn_queue_us / 1000U),
                 (unsigned)(latency_us / 1000U), (unsigned)full->overflows,
                 save ? " SAVE" : "");

        ++s_stats.windows;
        s_stats.feature_us_sum += feature_us;
        s_stats.invoke_us_sum += invoke_us;
        if (feature_us > s_stats.feature_us_max) {
            s_stats.feature_us_max = feature_us;
        }
        if (invoke_us > s_stats.invoke_us_max) {
            s_stats.invoke_us_max = invoke_us;
        }
        if ((s_stats.windows % STATS_PERIOD_WINDOWS) == 0U) {
            log_stats();
        }

        if (!save) {
            release_window(full->buffer);
            continue;
        }
        /* SD write goes to audio_writer; the window is returned after it is written. */
        save_job_t save_job = {.buffer = full->buffer, .window = full->window, .pred = pred};
        memcpy(save_job.probs, probs, sizeof(save_job.probs));
        (void)xQueueSend(s_save_jobs, &save_job, 0);
    }
}

/* ---------------------------------------------------------------------------
 * audio_writer (Core 1, lower priority): WAV + one CSV line per saved window
 * ------------------------------------------------------------------------- */

static void fill_wav_header(wav_header_t *hdr, uint32_t data_size)
{
    memcpy(hdr->riff, "RIFF", 4);
    memcpy(hdr->wave, "WAVE", 4);
    memcpy(hdr->fmt_, "fmt ", 4);
    memcpy(hdr->data, "data", 4);
    hdr->file_size_minus_8 = data_size + 36U;
    hdr->fmt_chunk_size    = 16U;
    hdr->audio_format      = 1U;    /* PCM */
    hdr->num_channels      = 1U;
    hdr->sample_rate       = MIC_SAMPLE_RATE;
    hdr->bits_per_sample   = 16U;
    hdr->byte_rate         = MIC_SAMPLE_RATE * sizeof(int16_t);
    hdr->block_align       = sizeof(int16_t);
    hdr->data_size         = data_size;
}

static esp_err_t write_all(int fd, const uint8_t *data, size_t bytes)
{
    size_t offset = 0;
    while (offset < bytes) {
        const ssize_t written = write(fd, data + offset, bytes - offset);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return ESP_FAIL;
        }
        offset += (size_t)written;
    }
    return ESP_OK;
}

/* The PSRAM window is staged through internal RAM in sector-sized chunks. */
static esp_err_t write_wav(const char *path, const int16_t *pcm)
{
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
        ESP_LOGE(TAG, "Cannot open %s: errno=%d", path, errno);
        return ESP_FAIL;
    }
    wav_header_t hdr = {0};
    fill_wav_header(&hdr, (uint32_t)WINDOW_BYTES);
    memcpy(s_write_buffer, &hdr, sizeof(hdr));
    size_t fill = sizeof(hdr);
    const uint8_t *source = (const uint8_t *)pcm;
    size_t remaining = WINDOW_BYTES;
    esp_err_t err = ESP_OK;
    while (remaining > 0U && err == ESP_OK) {
        const size_t room = WRITE_BUFFER_BYTES - fill;
        const size_t take = remaining < room ? remaining : room;
        memcpy(s_write_buffer + fill, source, take);
        source += take;
        remaining -= take;
        fill += take;
        err = write_all(fd, s_write_buffer, fill);
        fill = 0;
    }
    if (close(fd) != 0 && err == ESP_OK) {
        err = ESP_FAIL;
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "WAV write failed: %s errno=%d", path, errno);
        (void)unlink(path);
    }
    return err;
}

static void append_csv_line(const char *wav_name, const save_job_t *job)
{
    struct stat st;
    const bool new_file = stat(SD_LOG_PATH, &st) != 0 || st.st_size == 0;
    FILE *f = fopen(SD_LOG_PATH, "a");
    if (!f) {
        ESP_LOGW(TAG, "Cannot append %s", SD_LOG_PATH);
        return;
    }
    if (new_file) {
        fprintf(f, "file,window,pred,%s,%s,%s\n",
                g_audio_class_names[0], g_audio_class_names[1], g_audio_class_names[2]);
    }
    fprintf(f, "%s,%u,%s,%.4f,%.4f,%.4f\n", wav_name, (unsigned)job->window,
            g_audio_class_names[job->pred],
            (double)job->probs[0], (double)job->probs[1], (double)job->probs[2]);
    fclose(f);
}

static void audio_writer(void *arg)
{
    (void)arg;
    save_job_t job;
    char path[32];
    while (true) {
        (void)xQueueReceive(s_save_jobs, &job, portMAX_DELAY);
        wav_path_for_index(s_next_file_index, path, sizeof(path));
        gpio_set_level(DBG_SD_PIN, 1);
        const int64_t write_start_us = esp_timer_get_time();
        const esp_err_t err = write_wav(path, s_windows[job.buffer]);
        if (err == ESP_OK) {
            append_csv_line(path + sizeof(SD_MOUNT_POINT), &job);
        }
        const uint32_t write_ms = (uint32_t)((esp_timer_get_time() - write_start_us) / 1000);
        gpio_set_level(DBG_SD_PIN, 0);
        release_window(job.buffer);

        if (err == ESP_OK) {
            ++s_next_file_index;
            ++s_stats.saved;
            ESP_LOGI(TAG, "w%u saved %s (%u ms)", (unsigned)job.window, path, (unsigned)write_ms);
        } else {
            ++s_stats.save_failed;
        }
    }
}

/* ---------------------------------------------------------------------------
 * Startup
 * ------------------------------------------------------------------------- */

static esp_err_t pipeline_alloc(void)
{
    s_free_windows = xQueueCreateStatic(WINDOW_COUNT, sizeof(uint8_t),
                                        s_free_queue_storage, &s_free_queue_control);
    s_full_windows = xQueueCreateStatic(WINDOW_COUNT, sizeof(full_window_t),
                                        s_full_queue_storage, &s_full_queue_control);
    s_free_features = xQueueCreateStatic(FEATURE_COUNT, sizeof(uint8_t),
                                         s_free_feature_queue_storage,
                                         &s_free_feature_queue_control);
    s_feature_jobs = xQueueCreateStatic(FEATURE_COUNT, sizeof(feature_job_t),
                                        s_feature_queue_storage, &s_feature_queue_control);
    s_save_jobs = xQueueCreateStatic(WINDOW_COUNT, sizeof(save_job_t),
                                     s_save_queue_storage, &s_save_queue_control);
    if (!s_free_windows || !s_full_windows || !s_free_features || !s_feature_jobs ||
        !s_save_jobs) {
        return ESP_ERR_NO_MEM;
    }

    const uint32_t caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    for (uint8_t i = 0; i < WINDOW_COUNT; ++i) {
        s_windows[i] = heap_caps_malloc(WINDOW_BYTES, caps);
        memory_monitor_allocation("pcm_window", s_windows[i], WINDOW_BYTES);
        if (!s_windows[i]) {
            return ESP_ERR_NO_MEM;
        }
        (void)xQueueSend(s_free_windows, &i, 0);
    }
    s_feature_workspace_bytes = audio_logmel_int8_workspace_size();
    s_feature_workspace = heap_caps_aligned_alloc(16, s_feature_workspace_bytes, caps);
    memory_monitor_allocation("feature_workspace", s_feature_workspace,
                              s_feature_workspace_bytes);
    if (!s_feature_workspace) {
        return ESP_ERR_NO_MEM;
    }
    for (uint8_t i = 0; i < FEATURE_COUNT; ++i) {
        s_features[i] = heap_caps_aligned_alloc(16, AUDIO_LOGMEL_MODEL_INPUT_VALUES, caps);
        memory_monitor_allocation("model_input", s_features[i], AUDIO_LOGMEL_MODEL_INPUT_VALUES);
        if (!s_features[i]) {
            return ESP_ERR_NO_MEM;
        }
        (void)xQueueSend(s_free_features, &i, 0);
    }
    return ESP_OK;
}

void app_main(void)
{
    memory_monitor_log_capacities();
    memory_monitor_checkpoint("BOOT_ENTRY");
    dbg_pin_init();

    /* SD is optional: without it the stream still runs, it only skips saving. */
    if (STREAM_SAVE_MODE != SAVE_NONE) {
        s_sd_ready = sd_card_init() == ESP_OK;
        if (s_sd_ready) {
            s_next_file_index = find_next_file_index();
            ESP_LOGI(TAG, "SD ready; next file %u", (unsigned)s_next_file_index);
        } else {
            ESP_LOGW(TAG, "SD unavailable: inference continues without saving WAVs");
        }
    }

    if (pipeline_alloc() != ESP_OK) {
        ESP_LOGE(TAG, "Pipeline buffer allocation failed");
        return;
    }
    if (model_runner_int8_init() != ESP_OK) {
        ESP_LOGE(TAG, "INT8 model initialization failed");
        return;
    }
    if (i2s_init() != ESP_OK || i2s_channel_enable(s_rx_handle) != ESP_OK) {
        ESP_LOGE(TAG, "I2S initialization failed");
        return;
    }
    memory_monitor_checkpoint("PIPELINE_READY");

    /* Resident static tasks: no heap use and no per-cycle fragmentation. */
    s_writer_task = xTaskCreateStaticPinnedToCore(
        audio_writer, "audio_writer", WRITER_TASK_STACK_BYTES, NULL, WRITER_TASK_PRIORITY,
        s_writer_stack, &s_writer_tcb, WRITER_TASK_CORE);
    s_cnn_task = xTaskCreateStaticPinnedToCore(
        cnn_task, "cnn_task", CNN_TASK_STACK_BYTES, NULL, CNN_TASK_PRIORITY,
        s_cnn_stack, &s_cnn_tcb, CNN_TASK_CORE);
    s_logmel_task = xTaskCreateStaticPinnedToCore(
        logmel_task, "logmel_task", LOGMEL_TASK_STACK_BYTES, NULL, LOGMEL_TASK_PRIORITY,
        s_logmel_stack, &s_logmel_tcb, LOGMEL_TASK_CORE);
    s_capture_task = xTaskCreateStaticPinnedToCore(
        mic_capture, "mic_capture", CAPTURE_TASK_STACK_BYTES, NULL, CAPTURE_TASK_PRIORITY,
        s_capture_stack, &s_capture_tcb, CAPTURE_TASK_CORE);
    if (!s_writer_task || !s_cnn_task || !s_logmel_task || !s_capture_task) {
        ESP_LOGE(TAG, "Task creation failed");
        return;
    }

    ESP_LOGI(TAG, "Streaming: %s mic %u Hz, gain x%d, %u-sample windows; "
             "capture core %d, logmel core %d, cnn core %d, writer core %d; "
             "save mode %d (threshold %.2f)",
             MIC_NAME, (unsigned)MIC_SAMPLE_RATE, PCM_GAIN, (unsigned)WINDOW_SAMPLES,
             CAPTURE_TASK_CORE, LOGMEL_TASK_CORE, CNN_TASK_CORE, WRITER_TASK_CORE,
             STREAM_SAVE_MODE, (double)LOW_CONF_THRESHOLD);
    /* app_main returns; the four resident tasks keep running. */
}
