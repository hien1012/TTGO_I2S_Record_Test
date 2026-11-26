/*
 * Batch host build of the firmware int8 log-mel frontend.
 *
 * Input : raw little-endian PCM16, N consecutive windows of
 *         AUDIO_LOGMEL_INPUT_SAMPLES samples each (N is derived from file size).
 * Output: N consecutive int8 model tensors of AUDIO_LOGMEL_MODEL_INPUT_VALUES
 *         values each, in the NHWC order the TFLite model expects.
 *
 * Compile against the exact logmel_features_int8.c / preprocess_config_int8.h
 * used by the firmware so the tensors match the device bit for bit.
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logmel_features_int8.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s input_pcm16_windows.bin output_int8_tensors.bin\n", argv[0]);
        return 2;
    }

    FILE *in = fopen(argv[1], "rb");
    if (in == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", argv[1], strerror(errno));
        return 3;
    }
    if (fseek(in, 0, SEEK_END) != 0) {
        fprintf(stderr, "cannot seek %s\n", argv[1]);
        fclose(in);
        return 3;
    }
    const long input_bytes = ftell(in);
    rewind(in);
    const long window_bytes = (long)(AUDIO_LOGMEL_INPUT_SAMPLES * sizeof(int16_t));
    if (input_bytes <= 0 || input_bytes % window_bytes != 0) {
        fprintf(stderr, "input size %ld is not a positive multiple of %ld bytes\n",
                input_bytes, window_bytes);
        fclose(in);
        return 4;
    }
    const long window_count = input_bytes / window_bytes;

    FILE *out = fopen(argv[2], "wb");
    if (out == NULL) {
        fprintf(stderr, "cannot write %s: %s\n", argv[2], strerror(errno));
        fclose(in);
        return 3;
    }

    int16_t *pcm = (int16_t *)malloc((size_t)window_bytes);
    int8_t *model_input = (int8_t *)malloc(AUDIO_LOGMEL_MODEL_INPUT_VALUES);
    const size_t workspace_bytes = audio_logmel_int8_workspace_size();
    void *workspace = malloc(workspace_bytes);
    if (pcm == NULL || model_input == NULL || workspace == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 5;
    }

    int status = 0;
    for (long window = 0; window < window_count; ++window) {
        if (fread(pcm, sizeof(int16_t), AUDIO_LOGMEL_INPUT_SAMPLES, in) !=
            AUDIO_LOGMEL_INPUT_SAMPLES) {
            fprintf(stderr, "short read at window %ld\n", window);
            status = 6;
            break;
        }
        const int result = audio_logmel_extract_quantized_pcm16(
            pcm, AUDIO_LOGMEL_INPUT_SAMPLES, model_input, workspace, workspace_bytes, NULL);
        if (result != AUDIO_LOGMEL_OK) {
            fprintf(stderr, "feature extraction failed at window %ld: %d\n", window, result);
            status = 7;
            break;
        }
        if (fwrite(model_input, 1, AUDIO_LOGMEL_MODEL_INPUT_VALUES, out) !=
            AUDIO_LOGMEL_MODEL_INPUT_VALUES) {
            fprintf(stderr, "write failed at window %ld\n", window);
            status = 8;
            break;
        }
    }

    free(workspace);
    free(model_input);
    free(pcm);
    fclose(out);
    fclose(in);
    if (status == 0) {
        printf("%ld windows\n", window_count);
    }
    return status;
}
