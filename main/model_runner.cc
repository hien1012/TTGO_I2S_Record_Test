/**
 * @file model_runner.cc
 * @brief Lifetime management and invocation wrapper for the float32 CNN.
 *
 * Initialization is lazy so audio capture and feature extraction can be tested
 * independently. The tensor arena is allocated once and intentionally retained
 * for all future captures. This module is single-task by design; add a mutex if
 * inference is ever called from more than one task.
 */
#include "model_runner.h"

#include <algorithm>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "memory_monitor.h"
#include "model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr char TAG[] = "MODEL";
/* Float32 Conv2D activations dominate memory; place the arena in PSRAM first. */
constexpr int kTensorArenaSize = 512 * 1024;

static uint8_t *s_tensor_arena = nullptr;
static const tflite::Model *s_model = nullptr;
static tflite::MicroInterpreter *s_interpreter = nullptr;
static TfLiteTensor *s_input = nullptr;
static TfLiteTensor *s_output = nullptr;

}  // namespace

extern "C" esp_err_t model_runner_init(void)
{
    /* Successful initialization is idempotent; persistent objects are reused. */
    if (s_interpreter != nullptr) {
        return ESP_OK;
    }

    memory_monitor_checkpoint("MODEL_INIT_BEGIN");
    s_model = tflite::GetModel(g_audio_model_tflite);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema %ld != supported %d",
                 static_cast<long>(s_model->version()), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    memory_monitor_checkpoint("TENSOR_ARENA_ALLOC_BEGIN");
    /* Prefer PSRAM because float32 activations exceed practical internal DRAM. */
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

    /*
     * Keep this minimal resolver synchronized with logmel_conversion_summary.json.
     * Missing operators fail AllocateTensors/Invoke; extra operators consume Flash.
     */
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

    /* Reject a model that does not match the compiled frontend contract. */
    s_input = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    if (s_input->type != kTfLiteFloat32 || s_output->type != kTfLiteFloat32) {
        ESP_LOGE(TAG, "Expected float32 input/output, got input=%d output=%d",
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

    ESP_LOGI(TAG, "TFLite model ready: model=%u bytes arena=%d used=%u bytes",
             static_cast<unsigned>(g_audio_model_tflite_len),
             kTensorArenaSize,
             static_cast<unsigned>(s_interpreter->arena_used_bytes()));
    memory_monitor_checkpoint("MODEL_INIT_END");
    return ESP_OK;
}

extern "C" esp_err_t model_runner_predict(
    const float normalized_features[AUDIO_LOGMEL_MODEL_INPUT_VALUES],
    float probabilities[AUDIO_MODEL_CLASS_COUNT],
    int *predicted_class)
{
    if (!normalized_features || !probabilities || !predicted_class) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t ret = model_runner_init();
    if (ret != ESP_OK) {
        return ret;
    }

    /* TFLM owns its tensor memory, so copy from the caller-owned feature buffer. */
    std::memcpy(s_input->data.f, normalized_features,
                AUDIO_LOGMEL_MODEL_INPUT_VALUES * sizeof(float));

    memory_monitor_checkpoint("TFLITE_INVOKE_BEGIN");
    const TfLiteStatus invoke_status = s_interpreter->Invoke();
    memory_monitor_checkpoint("TFLITE_INVOKE_END");
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return ESP_FAIL;
    }

    int best = 0;
    for (int i = 0; i < AUDIO_MODEL_CLASS_COUNT; i++) {
        probabilities[i] = s_output->data.f[i];
        if (probabilities[i] > probabilities[best]) {
            best = i;
        }
    }

    *predicted_class = best;
    return ESP_OK;
}
