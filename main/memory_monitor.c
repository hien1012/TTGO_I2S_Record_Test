/**
 * @file memory_monitor.c
 * @brief Serial-log instrumentation for heap, fragmentation, and task stack.
 *
 * This module is diagnostic infrastructure, not an allocator. Calls are placed
 * outside time-critical I2S read loops because formatted logging perturbs timing
 * and may itself use small runtime buffers. Checkpoint timing must not be used as
 * a pure DSP/inference benchmark.
 */
#include "memory_monitor.h"

#include <stdbool.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_memory_utils.h"
#include "esp_ota_ops.h"
#include "esp_partition.h"
#include "esp_psram.h"

static const char *TAG = "MEMORY";

typedef struct {
    bool valid;
    size_t internal_free;
    size_t executable_free;
    size_t psram_free;
} memory_snapshot_t;

static memory_snapshot_t s_previous;

static long free_delta(size_t current, size_t previous, bool previous_valid)
{
    return previous_valid ? (long)current - (long)previous : 0L;
}

void memory_monitor_log_capacities(void)
{
    uint32_t flash_bytes = 0;
    esp_err_t flash_result = esp_flash_get_size(NULL, &flash_bytes);
    const esp_partition_t *running = esp_ota_get_running_partition();
    const size_t psram_bytes = esp_psram_get_size();

    ESP_LOGI(TAG,
             "CAPACITY nominal_dram=%u nominal_iram=%u flash=%u psram=%u app_partition=%u",
             320U * 1024U,
             200U * 1024U,
             flash_result == ESP_OK ? (unsigned)flash_bytes : 0U,
             (unsigned)psram_bytes,
             running != NULL ? (unsigned)running->size : 0U);
}

void memory_monitor_checkpoint(const char *stage)
{
    /* Capability views can overlap; each line is interpreted independently. */
    multi_heap_info_t internal = {0};
    multi_heap_info_t executable = {0};
    multi_heap_info_t psram = {0};

    heap_caps_get_info(&internal, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    heap_caps_get_info(&executable, MALLOC_CAP_INTERNAL | MALLOC_CAP_EXEC);
    heap_caps_get_info(&psram, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    const char *label = stage != NULL ? stage : "unnamed";
    ESP_LOGI(TAG,
             "[%s] DRAM used=%u free=%u delta_free=%+ld min_free=%u largest=%u",
             label,
             (unsigned)internal.total_allocated_bytes,
             (unsigned)internal.total_free_bytes,
             free_delta(internal.total_free_bytes,
                        s_previous.internal_free,
                        s_previous.valid),
             (unsigned)internal.minimum_free_bytes,
             (unsigned)internal.largest_free_block);
    ESP_LOGI(TAG,
             "[%s] IRAM_EXEC used=%u free=%u delta_free=%+ld min_free=%u largest=%u",
             label,
             (unsigned)executable.total_allocated_bytes,
             (unsigned)executable.total_free_bytes,
             free_delta(executable.total_free_bytes,
                        s_previous.executable_free,
                        s_previous.valid),
             (unsigned)executable.minimum_free_bytes,
             (unsigned)executable.largest_free_block);
    ESP_LOGI(TAG,
             "[%s] PSRAM used=%u free=%u delta_free=%+ld min_free=%u largest=%u",
             label,
             (unsigned)psram.total_allocated_bytes,
             (unsigned)psram.total_free_bytes,
             free_delta(psram.total_free_bytes,
                        s_previous.psram_free,
                        s_previous.valid),
             (unsigned)psram.minimum_free_bytes,
             (unsigned)psram.largest_free_block);
    /* ESP-IDF reports the minimum remaining stack in StackType_t units. */
    const UBaseType_t stack_hwm_words = uxTaskGetStackHighWaterMark(NULL);
    ESP_LOGI(TAG,
             "[%s] TASK stack_hwm=%u words (%u bytes minimum remaining)",
             label,
             (unsigned)stack_hwm_words,
             (unsigned)(stack_hwm_words * sizeof(StackType_t)));

    s_previous.valid = true;
    s_previous.internal_free = internal.total_free_bytes;
    s_previous.executable_free = executable.total_free_bytes;
    s_previous.psram_free = psram.total_free_bytes;
}

void memory_monitor_allocation(const char *name,
                               const void *pointer,
                               size_t requested_bytes)
{
    const char *location = "NULL";
    size_t actual_bytes = 0;

    if (pointer != NULL) {
        actual_bytes = heap_caps_get_allocated_size((void *)pointer);
        if (esp_ptr_external_ram(pointer)) {
            location = "PSRAM";
        } else if (esp_ptr_internal(pointer)) {
            location = "INTERNAL";
        } else {
            location = "OTHER";
        }
    }

    ESP_LOGI(TAG,
             "ALLOC name=%s requested=%u actual=%u location=%s ptr=%p",
             name != NULL ? name : "unnamed",
             (unsigned)requested_bytes,
             (unsigned)actual_bytes,
             location,
             pointer);
}
