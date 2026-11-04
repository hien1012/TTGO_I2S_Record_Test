#pragma once

/**
 * @file memory_monitor.h
 * @brief Runtime checkpoints for ESP32 memory and stack handoff analysis.
 *
 * ESP32 capability heaps may overlap in the physical memory map. DRAM and
 * IRAM_EXEC values therefore must not be added together. Stage deltas identify
 * allocations; min_free and largest_free_block expose pressure/fragmentation.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Print detected Flash, PSRAM, and application partition capacities. */
void memory_monitor_log_capacities(void);

/**
 * @brief Record heap and current-task stack state at a named pipeline stage.
 * @param[in] stage Stable label used to compare successive captures.
 *
 * Print heap state and the free-byte delta from the preceding checkpoint.
 * Internal DRAM, executable internal heap, and external PSRAM are reported
 * separately so stage-to-stage allocation changes are visible in serial logs.
 */
void memory_monitor_checkpoint(const char *stage);

/**
 * @brief Report allocation size and whether the pointer is internal or PSRAM.
 * @param[in] name Human-readable buffer name.
 * @param[in] pointer ESP-IDF heap pointer; may be NULL after allocation failure.
 * @param[in] requested_bytes Requested payload size in bytes.
 */
void memory_monitor_allocation(const char *name,
                               const void *pointer,
                               size_t requested_bytes);

#ifdef __cplusplus
}
#endif
