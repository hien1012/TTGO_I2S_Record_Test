#pragma once

/**
 * @file model_runner_int8.h
 * @brief C ABI for full-int8 TensorFlow Lite Micro inference.
 *
 * Input and output tensors are int8. Output values are dequantized to float
 * only for the application API and human-readable logs. The API is not
 * reentrant because the interpreter and tensor arena are singletons.
 */

#include "esp_err.h"
#include "logmel_features_int8.h"
#include "preprocess_config_int8.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Validate the int8 contract and allocate persistent TFLM state. */
esp_err_t model_runner_int8_init(void);

/**
 * @brief Invoke the int8 CNN and return dequantized class probabilities.
 * @param[in] quantized_features Exact NHWC tensor from the int8 frontend.
 * @param[out] probabilities Dequantized probabilities in generated class order.
 * @param[out] predicted_class Index of the largest probability.
 * @return ESP_OK on success; an ESP-IDF error otherwise.
 */
esp_err_t model_runner_int8_predict(
    const int8_t quantized_features[AUDIO_LOGMEL_MODEL_INPUT_VALUES],
    float probabilities[AUDIO_MODEL_CLASS_COUNT],
    int *predicted_class);

#ifdef __cplusplus
}
#endif
