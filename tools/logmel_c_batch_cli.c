/*
 * Batch host runner for the firmware log-Mel CMVN frontend (main/logmel_features.c).
 *
 * Input:  count x 32000 little-endian int16 samples (2 s at 16 kHz).
 * Output: count x (32 x 124) float32 per-band CMVN values in [mel][frame] order,
 *         taken before the deployed model's training-statistic normalization so
 *         a newly trained model can apply its own statistics.
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logmel_features.h"

#if AUDIO_MODEL_FEATURE_MODE != AUDIO_MODEL_FEATURE_CMVN
#error "logmel_c_batch_cli expects preprocess_config.h to select CMVN features"
#endif

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s input_pcm16.bin output_f32.bin clip_count\n", argv[0]);
        return 2;
    }
    size_t count = strtoul(argv[3], NULL, 10);
    FILE *in = fopen(argv[1], "rb");
    FILE *out = fopen(argv[2], "wb");
    if (in == NULL || out == NULL) {
        fprintf(stderr, "cannot open input/output: %s\n", strerror(errno));
        return 3;
    }

    int16_t *pcm = (int16_t *)malloc(AUDIO_LOGMEL_INPUT_SAMPLES * sizeof(int16_t));
    float *model_input = (float *)malloc(AUDIO_LOGMEL_MODEL_INPUT_VALUES * sizeof(float));
    float *cmvn = (float *)malloc(AUDIO_LOGMEL_PLANE_VALUES * sizeof(float));
    /* One workspace for every clip, matching the firmware's single PSRAM workspace. */
    void *workspace = malloc(audio_logmel_workspace_size());
    if (pcm == NULL || model_input == NULL || cmvn == NULL || workspace == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 4;
    }
    audio_logmel_debug_buffers_t debug = {0};
    debug.cmvn = cmvn;

    for (size_t clip = 0; clip < count; ++clip) {
        if (fread(pcm, sizeof(int16_t), AUDIO_LOGMEL_INPUT_SAMPLES, in) !=
            AUDIO_LOGMEL_INPUT_SAMPLES) {
            fprintf(stderr, "short read at clip %u\n", (unsigned)clip);
            return 5;
        }
        int result = audio_logmel_extract_normalized_pcm16(
            pcm,
            AUDIO_LOGMEL_INPUT_SAMPLES,
            model_input,
            workspace,
            audio_logmel_workspace_size(),
            &debug);
        if (result != AUDIO_LOGMEL_OK) {
            fprintf(stderr, "feature extraction failed at clip %u: %d\n", (unsigned)clip, result);
            return 6;
        }
        if (fwrite(cmvn, sizeof(float), AUDIO_LOGMEL_PLANE_VALUES, out) !=
            AUDIO_LOGMEL_PLANE_VALUES) {
            fprintf(stderr, "short write at clip %u\n", (unsigned)clip);
            return 7;
        }
    }

    free(workspace);
    free(cmvn);
    free(model_input);
    free(pcm);
    fclose(in);
    fclose(out);
    return 0;
}
