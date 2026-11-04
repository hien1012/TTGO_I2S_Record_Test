#pragma once

/**
 * @file model_runner.h
 * @brief C ABI for lazy float32 TensorFlow Lite Micro inference.
 *
 * Interpreter and tensor-arena objects live for the process lifetime. This API
 * is not reentrant and must not be invoked concurrently by multiple tasks.
 */

#include "esp_err.h"
#include "logmel_features.h"
#include "preprocess_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Validate the embedded model and allocate persistent TFLM state.
 * @return ESP_OK when ready; an ESP-IDF error otherwise.
 * @note Repeated successful calls are no-ops.
 */
esp_err_t model_runner_init(void);

/**
 * @brief Run one float32 inference.
 * @param[in] normalized_features NHWC input produced by logmel_features.
 * @param[out] probabilities Softmax probabilities in generated class order.
 * @param[out] predicted_class Index of the largest probability.
 * @return ESP_OK on success; initialization or Invoke error otherwise.
 */
esp_err_t model_runner_predict(
                               const float normalized_features[AUDIO_LOGMEL_MODEL_INPUT_VALUES],
                               float probabilities[AUDIO_MODEL_CLASS_COUNT],
                               int *predicted_class);

#ifdef __cplusplus
}
#endif
