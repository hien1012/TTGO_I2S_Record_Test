#pragma once

/**
 * @file logmel_features.h
 * @brief Float32 log-Mel CMVN frontend shared by capture and TFLM inference.
 *
 * These constants form a deployment contract with the Python training pipeline
 * and embedded model. Changes to sample rate, frame geometry, Mel count,
 * channel mode, or normalization require retraining and parity validation.
 *
 * Data layout:
 *   - Input:  32000 mono PCM16 samples (2 s at 16 kHz).
 *   - Output: NHWC values ordered [mel][frame][channel].
 *   - Current CMVN model: 32 * 124 * 1 float32 values.
 *
 * This module performs no heap allocation. The caller owns every buffer.
 */

#include <stddef.h>
#include <stdint.h>

#include "preprocess_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed dimensions. Keep aligned with the generated preprocessing contract. */
#define AUDIO_LOGMEL_SAMPLE_RATE_HZ      16000
#define AUDIO_LOGMEL_RECORD_SECONDS      2
#define AUDIO_LOGMEL_INPUT_SAMPLES       32000
#define AUDIO_LOGMEL_N_FFT               512
#define AUDIO_LOGMEL_HOP_LENGTH          256
#define AUDIO_LOGMEL_FFT_BINS            257
#define AUDIO_LOGMEL_N_MELS              32
#define AUDIO_LOGMEL_FRAME_COUNT         124
#define AUDIO_LOGMEL_CHANNEL_COUNT       AUDIO_MODEL_INPUT_CHANNELS
#define AUDIO_LOGMEL_PLANE_VALUES        (AUDIO_LOGMEL_N_MELS * AUDIO_LOGMEL_FRAME_COUNT)
#define AUDIO_LOGMEL_MODEL_INPUT_VALUES  (AUDIO_LOGMEL_PLANE_VALUES * AUDIO_LOGMEL_CHANNEL_COUNT)

enum {
    AUDIO_LOGMEL_OK = 0,
    AUDIO_LOGMEL_ERR_INVALID_ARG = -1,
    AUDIO_LOGMEL_ERR_BAD_SAMPLE_COUNT = -2,
    AUDIO_LOGMEL_ERR_WORKSPACE_TOO_SMALL = -3,
};

/**
 * Optional buffers for Python/C stage comparison. All pointers may be NULL.
 * power_spectrogram uses [frame][fft_bin]; Mel arrays use [mel][frame].
 * normalized_input uses model NHWC order [mel][frame][channel].
 *
 * @note Enabling all debug buffers consumes substantially more RAM than the
 *       production inference path, so production callers should pass NULL.
 */
typedef struct {
    float *dc_removed_audio;       /* 32000 floats */
    float *power_spectrogram;      /* 124 * 257 floats */
    float *mel_power;              /* 32 * 124 floats */
    float *log_mel;                /* 32 * 124 floats */
    float *delta;                  /* 32 * 124 floats */
    float *cmvn;                   /* Optional 32 * 124 CMVN values */
    float *normalized_input;       /* 32 * 124 * model channel count */
} audio_logmel_debug_buffers_t;

/** @return Exact extractor workspace requirement in bytes. */
size_t audio_logmel_workspace_size(void);

/**
 * @brief Convert one fixed PCM16 capture into normalized float32 model input.
 *
 * Processing order: PCM scaling, utterance DC removal, Hamming window, radix-2
 * FFT, power spectrum, HTK Mel filters, dB compression, top-dB clipping,
 * optional per-Mel CMVN, and training-statistic normalization.
 *
 * @param[in] pcm Signed mono samples used for both WAV storage and inference.
 * @param[in] sample_count Must equal AUDIO_LOGMEL_INPUT_SAMPLES.
 * @param[out] model_input Caller-owned NHWC float32 tensor.
 * @param[in,out] workspace Caller-owned, 4-byte-aligned DSP scratch region.
 * @param[in] workspace_bytes Size of workspace in bytes.
 * @param[out] debug_buffers Optional parity outputs; pass NULL in production.
 * @return AUDIO_LOGMEL_OK on success, otherwise AUDIO_LOGMEL_ERR_*.
 *
 * Input must be exactly two seconds of mono 16 kHz PCM16. The returned model
 * input is float32 NHWC data and is already normalized with training-only
 * statistics generated in preprocess_config.h.
 */
int audio_logmel_extract_normalized_pcm16(
    const int16_t *pcm,
    size_t sample_count,
    float model_input[AUDIO_LOGMEL_MODEL_INPUT_VALUES],
    void *workspace,
    size_t workspace_bytes,
    const audio_logmel_debug_buffers_t *debug_buffers);

#ifdef __cplusplus
}
#endif
