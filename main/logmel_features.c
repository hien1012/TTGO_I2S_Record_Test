/**
 * @file logmel_features.c
 * @brief Deterministic float32 implementation of the training log-Mel frontend.
 *
 * Design constraints:
 *   - Match compare_logmel_python_c.py within the documented tolerance.
 *   - Keep large scratch arrays in a caller-provided PSRAM workspace.
 *   - Avoid per-frame allocation and hidden persistent state.
 *   - Preserve [mel][time][channel] order expected by the TFLite model.
 *
 * The hand-written radix-2 FFT favors auditability and Python parity. Replacing
 * it with an optimized DSP library is permitted only after stage-by-stage
 * numerical comparison and on-device timing measurements.
 */
#include "logmel_features.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "preprocess_config.h"

#if AUDIO_MODEL_INPUT_HEIGHT != AUDIO_LOGMEL_N_MELS || \
    AUDIO_MODEL_INPUT_WIDTH != AUDIO_LOGMEL_FRAME_COUNT || \
    AUDIO_MODEL_INPUT_CHANNELS != AUDIO_LOGMEL_CHANNEL_COUNT || \
    AUDIO_MODEL_INPUT_VALUES != AUDIO_LOGMEL_MODEL_INPUT_VALUES
#error "Generated model input contract does not match logmel_features.h"
#endif

#if AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_LOGMEL
#if AUDIO_MODEL_INPUT_CHANNELS != 1
#error "Raw log-mel models require exactly one input channel"
#endif
#elif AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_LOGMEL_DELTA
#if AUDIO_MODEL_INPUT_CHANNELS != 2
#error "Log-mel+delta models require exactly two input channels"
#endif
#elif AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_CMVN
#if AUDIO_MODEL_INPUT_CHANNELS != 1
#error "CMVN log-mel models require exactly one input channel"
#endif
#else
#error "Unsupported generated log-mel feature mode"
#endif

#ifndef AUDIO_MODEL_MEL_F_MIN_HZ
#error "preprocess_config.h lacks the mel band; regenerate it with tools/convert_logmel_tflite_to_c.py"
#endif

#define AUDIO_LOG_FLOOR 1.0e-12f
#define AUDIO_LOG_TOP_DB 80.0f
#define AUDIO_PI 3.14159265358979323846

/* All mutable DSP state is grouped so one allocation can be placed in PSRAM. */
typedef struct {
    float fft_re[AUDIO_LOGMEL_N_FFT];
    float fft_im[AUDIO_LOGMEL_N_FFT]; // 幾乎不會用到虛部
    float power[AUDIO_LOGMEL_FFT_BINS];
    float hamming[AUDIO_LOGMEL_N_FFT];
    float twiddle_cos[AUDIO_LOGMEL_N_FFT / 2];
    float twiddle_sin[AUDIO_LOGMEL_N_FFT / 2];
    float log_mel[AUDIO_LOGMEL_PLANE_VALUES];
} audio_logmel_workspace_t;

static void initialize_tables(audio_logmel_workspace_t *work)
{
    static bool initialized = false;  // 加flag
    if (initialized) return;          // 已算過就直接跳過
    initialized = true;               // 否則 set flag

    /* Tables are regenerated per call to keep this module stateless/reentrant. */

    // Hamming 窗：512 個 cos 值
    const double denominator = (double)(AUDIO_LOGMEL_N_FFT - 1); 
    for (int i = 0; i < AUDIO_LOGMEL_N_FFT; ++i) {               // i = 0,1,2,..,511
        double phase = 2.0 * AUDIO_PI * (double)i / denominator; //2π × i / 511
        work->hamming[i] = (float)(0.54 - 0.46 * cos(phase));    //0.54 - 0.46 × cos(2πi / 511)
    }

    // FFT 旋轉因子：256 個 cos/sin 值
    for (int i = 0; i < AUDIO_LOGMEL_N_FFT / 2; ++i) {          // i = 0,1,2,..,255
        double phase = 2.0 * AUDIO_PI * (double)i / (double)AUDIO_LOGMEL_N_FFT; // 2πi / 512
        work->twiddle_cos[i] = (float)cos(phase);               // cos(2πi / 512)
        work->twiddle_sin[i] = (float)sin(phase);               // sin(2πi / 512)
    }

    // Mel 濾波器邊界不在此計算：converter 依訓練時的 f_min/f_max，
    // 用 Python 同一個函式算出 34 個 FFT bin index，寫進 preprocess_config.h
    // 的 g_audio_mel_bin_edges，確保與訓練逐 bin 一致。
}

static void fft_radix2(audio_logmel_workspace_t *work)
{
    /* In-place decimation-in-time FFT; N_FFT must remain a power of two. */
    int j = 0;
    for (int i = 1; i < AUDIO_LOGMEL_N_FFT; ++i) {
        int bit = AUDIO_LOGMEL_N_FFT >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            float tmp = work->fft_re[i];
            work->fft_re[i] = work->fft_re[j];
            work->fft_re[j] = tmp;
            tmp = work->fft_im[i];
            work->fft_im[i] = work->fft_im[j];
            work->fft_im[j] = tmp;
        }
    }

    for (int length = 2; length <= AUDIO_LOGMEL_N_FFT; length <<= 1) {
        int half = length >> 1;
        int twiddle_step = AUDIO_LOGMEL_N_FFT / length;
        for (int start = 0; start < AUDIO_LOGMEL_N_FFT; start += length) {
            for (int offset = 0; offset < half; ++offset) {
                int twiddle_index = offset * twiddle_step;
                float wr = work->twiddle_cos[twiddle_index];
                float wi = -work->twiddle_sin[twiddle_index];
                int even = start + offset;
                int odd = even + half;
                float odd_re = work->fft_re[odd];
                float odd_im = work->fft_im[odd];
                float tr = wr * odd_re - wi * odd_im;
                float ti = wr * odd_im + wi * odd_re;
                float even_re = work->fft_re[even];
                float even_im = work->fft_im[even];
                work->fft_re[even] = even_re + tr;
                work->fft_im[even] = even_im + ti;
                work->fft_re[odd] = even_re - tr;
                work->fft_im[odd] = even_im - ti;
            }
        }
    }
}

static float mel_filter_power(const audio_logmel_workspace_t *work, int mel_index)
{
    /* Apply one triangular filter using generated FFT-bin boundaries. */
    int left = g_audio_mel_bin_edges[mel_index];
    int center = g_audio_mel_bin_edges[mel_index + 1];
    int right = g_audio_mel_bin_edges[mel_index + 2];
    float sum = 0.0f;

    if (center > left) {
        float width = (float)(center - left);
        for (int bin = left; bin < center; ++bin) {
            float weight = (float)(bin - left) / width;
            sum += weight * work->power[bin];
        }
    }
    if (right > center) {
        float width = (float)(right - center);
        for (int bin = center; bin < right; ++bin) {
            float weight = 1.0f - (float)(bin - center) / width;
            sum += weight * work->power[bin];
        }
    }
    return sum > 0.0f ? sum : 0.0f;
}

size_t audio_logmel_workspace_size(void)
{
    return sizeof(audio_logmel_workspace_t);
}

int audio_logmel_extract_normalized_pcm16(
    const int16_t *pcm,                                     // 輸入：32000 個 int16_t
    size_t sample_count,                                    // 必須 == 32000
    float model_input[AUDIO_LOGMEL_MODEL_INPUT_VALUES],     // 輸出：TFLite 輸入張量
    void *workspace,                                        // DSP 工作區（PSRAM）
    size_t workspace_bytes,                                 // sizeof(audio_logmel_workspace_t);
    const audio_logmel_debug_buffers_t *debug_buffers)      // 通常 NULL
{
    if (pcm == NULL || model_input == NULL || workspace == NULL) {
        return AUDIO_LOGMEL_ERR_INVALID_ARG;
    }
    if (sample_count != AUDIO_LOGMEL_INPUT_SAMPLES) {
        return AUDIO_LOGMEL_ERR_BAD_SAMPLE_COUNT;
    }
    if (workspace_bytes < sizeof(audio_logmel_workspace_t)) {
        return AUDIO_LOGMEL_ERR_WORKSPACE_TOO_SMALL;
    }
    // step 0 
    audio_logmel_workspace_t *work = (audio_logmel_workspace_t *)workspace;
    initialize_tables(work);

    /* Phase 1: compute one utterance-level DC offset before framing. */
    double sum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        sum += (double)pcm[i] / 32767.0;
    }
    const float dc_mean = (float)(sum / (double)sample_count);
    float global_log_max = -INFINITY;

    /* Phase 2: produce an unclipped log-Mel plane and its global dB reference. */
    /* 逐幀處理（124 次迴圈）: 每幀從 pcm_buffer 裡切出 512 個 sample，向右滑動 256 個： */

    
    for (int frame = 0; frame < AUDIO_LOGMEL_FRAME_COUNT; ++frame) { //每一 frame 512 bit
        /* Step 2a：加窗（Frame + Hamming window） */
        int start = frame * AUDIO_LOGMEL_HOP_LENGTH;
        // frame & windowing
        for (int i = 0; i < AUDIO_LOGMEL_N_FFT; ++i) { // for every sample point in a frame
            float centered = (float)pcm[start + i] / 32767.0f - dc_mean; // subtract dc offset
            work->fft_re[i] = centered * work->hamming[i];               // 加窗 multiply with window
            work->fft_im[i] = 0.0f;                                      // 虛部清零
        }
        /* Step 2b : compute FFT : // 原地計算，結果在 fft_re[] / fft_im[] */
        fft_radix2(work); // 原地計算，結果在 fft_re[] / fft_im[]

        //Step 2c：計算功率頻譜 every FFT bins (512 but symmetric so just take positive part + 1 = 257)
        for (int bin = 0; bin < AUDIO_LOGMEL_FFT_BINS; ++bin) {     /* 每個 bin 代表的頻率：                              */
            float re = work->fft_re[bin];                           /* bin 0   → 0 Hz（DC）, bin 1   → 16000/512 ≈ 31.25 */
            float im = work->fft_im[bin];                           /* Hz bin 128 → 4000 Hz, bin 256 → 8000 Hz（Nyquist) */
            float value = re * re + im * im; // only re^2, im^2 = 0
            work->power[bin] = value;
            if (debug_buffers != NULL && debug_buffers->power_spectrogram != NULL) {
                debug_buffers->power_spectrogram[frame * AUDIO_LOGMEL_FFT_BINS + bin] = value;
            }
        }

        // Step 2d：套用 32 個 Mel 三角濾波器 apply 32 mel filter for every frame
        for (int mel = 0; mel < AUDIO_LOGMEL_N_MELS; ++mel) {
            float mel_power = mel_filter_power(work, mel);
            int index = mel * AUDIO_LOGMEL_FRAME_COUNT + frame;
            if (debug_buffers != NULL && debug_buffers->mel_power != NULL) {
                debug_buffers->mel_power[index] = mel_power;
            }
            float log_value = 10.0f * log10f(
                mel_power > AUDIO_LOG_FLOOR ? mel_power : AUDIO_LOG_FLOOR);
            work->log_mel[index] = log_value;
            if (log_value > global_log_max) {
                global_log_max = log_value;
            }
        }
    }

    if (debug_buffers != NULL && debug_buffers->dc_removed_audio != NULL) {
        for (size_t i = 0; i < sample_count; ++i) {
            debug_buffers->dc_removed_audio[i] =
                (float)pcm[i] / 32767.0f - dc_mean;
        }
    }

    /*
     * Phase 3: clip relative dB values, compute per-band CMVN where selected,
     * and write the final normalized NHWC tensor without another large buffer.
     */
    for (int mel = 0; mel < AUDIO_LOGMEL_N_MELS; ++mel) {  // for every mel band
#if AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_CMVN
        /* Python uses population statistics over time for each Mel band. */

        // compute mean of a mel band ( accross 124 frame)
        float cmvn_sum = 0.0f;
        for (int frame = 0; frame < AUDIO_LOGMEL_FRAME_COUNT; ++frame) {
            int plane_index = mel * AUDIO_LOGMEL_FRAME_COUNT + frame;
            float value = work->log_mel[plane_index] - global_log_max;
            if (value < -AUDIO_LOG_TOP_DB) {
                value = -AUDIO_LOG_TOP_DB;
            }
            cmvn_sum += value;
        }
        const float cmvn_mean = cmvn_sum / (float)AUDIO_LOGMEL_FRAME_COUNT;

        // compute standard deviation of a mel band ( accross 124 frame)
        float cmvn_squared_sum = 0.0f;
        for (int frame = 0; frame < AUDIO_LOGMEL_FRAME_COUNT; ++frame) {
            int plane_index = mel * AUDIO_LOGMEL_FRAME_COUNT + frame;
            float value = work->log_mel[plane_index] - global_log_max;
            if (value < -AUDIO_LOG_TOP_DB) {
                value = -AUDIO_LOG_TOP_DB;
            }
            float difference = value - cmvn_mean;
            cmvn_squared_sum += difference * difference;
        }
        float cmvn_std = sqrtf(cmvn_squared_sum / (float)AUDIO_LOGMEL_FRAME_COUNT);
        if (cmvn_std < AUDIO_MODEL_CMVN_STD_FLOOR_DB) {
            cmvn_std = AUDIO_MODEL_CMVN_STD_FLOOR_DB;
        }
        const float cmvn_inv_std = 1.0f / cmvn_std;
#endif

        // clip the value from -80db ~ 0db
        float previous = 0.0f;
        for (int frame = 0; frame < AUDIO_LOGMEL_FRAME_COUNT; ++frame) {
            int plane_index = mel * AUDIO_LOGMEL_FRAME_COUNT + frame;
            float log_value = work->log_mel[plane_index] - global_log_max;
            if (log_value < -AUDIO_LOG_TOP_DB) {
                log_value = -AUDIO_LOG_TOP_DB;
            }
            float delta = frame == 0 ? 0.0f : log_value - previous;
            previous = log_value;

#if AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_LOGMEL
            model_input[plane_index] =
                (log_value - g_audio_input_mean[0]) * g_audio_input_inv_std[0];
#elif AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_LOGMEL_DELTA
            int input_index = plane_index * AUDIO_LOGMEL_CHANNEL_COUNT;
            model_input[input_index] =
                (log_value - g_audio_input_mean[0]) * g_audio_input_inv_std[0];
            model_input[input_index + 1] =
                (delta - g_audio_input_mean[1]) * g_audio_input_inv_std[1];
                
/* compute cmvn of every point [124,32] */
#elif AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_CMVN
            // compute cmvn value of every point in [124, 32] 
            float cmvn_value = (log_value - cmvn_mean) * cmvn_inv_std;
            model_input[plane_index] =
                (cmvn_value - g_audio_input_mean[0]) * g_audio_input_inv_std[0];
#endif

            if (debug_buffers != NULL) {
                if (debug_buffers->log_mel != NULL) {
                    debug_buffers->log_mel[plane_index] = log_value;
                }
                if (debug_buffers->delta != NULL) {
                    debug_buffers->delta[plane_index] = delta;
                }
#if AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_CMVN
                if (debug_buffers->cmvn != NULL) {
                    debug_buffers->cmvn[plane_index] = cmvn_value;
                }
#endif
            }
        }
    }

    if (debug_buffers != NULL && debug_buffers->normalized_input != NULL) {
        memcpy(debug_buffers->normalized_input,
               model_input,
               AUDIO_LOGMEL_MODEL_INPUT_VALUES * sizeof(float));
    }
    return AUDIO_LOGMEL_OK;
}
