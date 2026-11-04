/**
 * @file model_runner_int8.cc
 * @brief Lifetime management and invocation wrapper for the full-int8 CNN.
 *
 * The embedded model, generated scale/zero-point constants, and frontend output
 * are validated as one contract before inference. The tensor arena is retained
 * for process lifetime and the module is intentionally non-reentrant.
 */
#include "model_runner_int8.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "memory_monitor.h"
#include "model_data_int8.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr char TAG[] = "MODEL_INT8";
/* Start with margin above the expected int8 activations; verify on hardware. */
constexpr int kTensorArenaSize = 256 * 1024;
constexpr float kScaleTolerance = 1.0e-7f;

static uint8_t *s_tensor_arena = nullptr;
static const tflite::Model *s_model = nullptr;
static tflite::MicroInterpreter *s_interpreter = nullptr;
static TfLiteTensor *s_input = nullptr;
static TfLiteTensor *s_output = nullptr;

}  // namespace

extern "C" esp_err_t model_runner_int8_init(void)
{
    /* Reuse the interpreter and arena after the first successful initialization. */
    if (s_interpreter != nullptr) {
        return ESP_OK;
    }

    memory_monitor_checkpoint("MODEL_INIT_BEGIN");
    s_model = tflite::GetModel(g_audio_model_int8_tflite);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema %ld != supported %d",
                 static_cast<long>(s_model->version()), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    /* Prefer PSRAM; internal fallback is retained for boards with sufficient RAM. */
    memory_monitor_checkpoint("TENSOR_ARENA_ALLOC_BEGIN");
    s_tensor_arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16,
        kTensorArenaSize,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (s_tensor_arena == nullptr) {
        s_tensor_arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(
            16,
            kTensorArenaSize,
            MALLOC_CAP_8BIT));
    }
    if (s_tensor_arena == nullptr) {
        ESP_LOGE(TAG, "tensor arena alloc failed: %d bytes", kTensorArenaSize);
        return ESP_ERR_NO_MEM;
    }
    memory_monitor_allocation("tensor_arena", s_tensor_arena, kTensorArenaSize);
    memory_monitor_checkpoint("TENSOR_ARENA_ALLOC_END");

    /* Keep the resolver synchronized with logmel_conversion_summary_int8.json. */
    static tflite::MicroMutableOpResolver<6> resolver;
    if (resolver.AddAveragePool2D() != kTfLiteOk) {
        ESP_LOGE(TAG, "AddAveragePool2D failed");
        return ESP_FAIL;
    }
    if (resolver.AddConv2D() != kTfLiteOk) {
        ESP_LOGE(TAG, "AddConv2D failed");
        return ESP_FAIL;
    }
    if (resolver.AddDepthwiseConv2D() != kTfLiteOk) {
        ESP_LOGE(TAG, "AddDepthwiseConv2D failed");
        return ESP_FAIL;
    }
    if (resolver.AddMaxPool2D() != kTfLiteOk) {
        ESP_LOGE(TAG, "AddMaxPool2D failed");
        return ESP_FAIL;
    }
    if (resolver.AddMean() != kTfLiteOk) {
        ESP_LOGE(TAG, "AddMean failed");
        return ESP_FAIL;
    }
    if (resolver.AddSoftmax() != kTfLiteOk) {
        ESP_LOGE(TAG, "AddSoftmax failed");
        return ESP_FAIL;
    }

    static tflite::MicroInterpreter interpreter(
        s_model,
        resolver,
        s_tensor_arena,
        kTensorArenaSize);

    s_interpreter = &interpreter;

    memory_monitor_checkpoint("ALLOCATE_TENSORS_BEGIN");
    const TfLiteStatus allocate_status = s_interpreter->AllocateTensors();
    memory_monitor_checkpoint("ALLOCATE_TENSORS_END");
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return ESP_FAIL;
    }

    s_input = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    /* Full-int8 deployment forbids float input/output fallback. */
    if (s_input->type != kTfLiteInt8 || s_output->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Expected int8 input/output, got input=%d output=%d",
                 s_input->type, s_output->type);
        return ESP_ERR_INVALID_STATE;
    }
    if (s_input->dims->size != 4 ||
        s_input->dims->data[0] != 1 ||
        s_input->dims->data[1] != AUDIO_LOGMEL_N_MELS ||
        s_input->dims->data[2] != AUDIO_LOGMEL_FRAME_COUNT ||
        s_input->dims->data[3] != AUDIO_LOGMEL_CHANNEL_COUNT) {
        ESP_LOGE(TAG, "Unexpected input shape");
        return ESP_ERR_INVALID_SIZE;
    }
    if (s_output->dims->size != 2 ||
        s_output->dims->data[0] != 1 ||
        s_output->dims->data[1] != AUDIO_MODEL_CLASS_COUNT) {
        ESP_LOGE(TAG, "Unexpected output shape");
        return ESP_ERR_INVALID_SIZE;
    }

    /* Catch stale generated headers before they silently corrupt predictions. */
    if (std::fabs(s_input->params.scale - AUDIO_MODEL_INPUT_SCALE) > kScaleTolerance ||
        s_input->params.zero_point != AUDIO_MODEL_INPUT_ZERO_POINT ||
        std::fabs(s_output->params.scale - AUDIO_MODEL_OUTPUT_SCALE) > kScaleTolerance ||
        s_output->params.zero_point != AUDIO_MODEL_OUTPUT_ZERO_POINT) {
        ESP_LOGE(TAG, "Generated int8 quantization constants do not match model tensors");
        return ESP_ERR_INVALID_STATE;
    }

    ESP_LOGI(TAG, "TFLite int8 model ready: model=%u bytes arena=%d used=%u bytes",
             static_cast<unsigned>(g_audio_model_int8_tflite_len),
             kTensorArenaSize,
             static_cast<unsigned>(s_interpreter->arena_used_bytes()));
    memory_monitor_checkpoint("MODEL_INIT_END");
    return ESP_OK;
}

extern "C" esp_err_t model_runner_int8_predict(
    const int8_t quantized_features[AUDIO_LOGMEL_MODEL_INPUT_VALUES],
    float probabilities[AUDIO_MODEL_CLASS_COUNT],
    int *predicted_class)
{
    if (!quantized_features || !probabilities || !predicted_class) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t ret = model_runner_int8_init();
    if (ret != ESP_OK) {
        return ret;
    }

    /* Input bytes are already quantized by logmel_features_int8.c. */
    std::memcpy(s_input->data.int8, quantized_features,
                AUDIO_LOGMEL_MODEL_INPUT_VALUES * sizeof(int8_t));

    memory_monitor_checkpoint("TFLITE_INVOKE_BEGIN");
    const TfLiteStatus invoke_status = s_interpreter->Invoke();
    memory_monitor_checkpoint("TFLITE_INVOKE_END");
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return ESP_FAIL;
    }

    /* Dequantize only three output scores for readable application logging. */
    int best = 0;
    for (int i = 0; i < AUDIO_MODEL_CLASS_COUNT; i++) {
        probabilities[i] =
            (static_cast<int32_t>(s_output->data.int8[i]) -
             AUDIO_MODEL_OUTPUT_ZERO_POINT) * AUDIO_MODEL_OUTPUT_SCALE;
        if (probabilities[i] > probabilities[best]) {
            best = i;
        }
    }

    *predicted_class = best;
    return ESP_OK;
}
