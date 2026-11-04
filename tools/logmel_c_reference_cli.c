#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "logmel_features.h"

static int read_exact(const char *path, void *data, size_t item_size, size_t count)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    size_t actual = fread(data, item_size, count, file);
    int extra = fgetc(file);
    fclose(file);
    if (actual != count || extra != EOF) {
        fprintf(stderr, "unexpected input size: expected %zu items, got %zu\n", count, actual);
        return -1;
    }
    return 0;
}

static int write_floats(const char *out_dir, const char *name, const float *data, size_t count)
{
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/%s", out_dir, name);
    if (length <= 0 || (size_t)length >= sizeof(path)) {
        return -1;
    }
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "cannot write %s: %s\n", path, strerror(errno));
        return -1;
    }
    size_t actual = fwrite(data, sizeof(float), count, file);
    fclose(file);
    return actual == count ? 0 : -1;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s input_pcm16.bin output_directory\n", argv[0]);
        return 2;
    }

    int16_t *pcm = (int16_t *)malloc(AUDIO_LOGMEL_INPUT_SAMPLES * sizeof(int16_t));
    float *model_input =
        (float *)malloc(AUDIO_LOGMEL_MODEL_INPUT_VALUES * sizeof(float));
    void *workspace = malloc(audio_logmel_workspace_size());
    audio_logmel_debug_buffers_t debug = {
        .dc_removed_audio = (float *)malloc(AUDIO_LOGMEL_INPUT_SAMPLES * sizeof(float)),
        .power_spectrogram = (float *)malloc(
            AUDIO_LOGMEL_FRAME_COUNT * AUDIO_LOGMEL_FFT_BINS * sizeof(float)),
        .mel_power = (float *)malloc(AUDIO_LOGMEL_PLANE_VALUES * sizeof(float)),
        .log_mel = (float *)malloc(AUDIO_LOGMEL_PLANE_VALUES * sizeof(float)),
        .delta = (float *)malloc(AUDIO_LOGMEL_PLANE_VALUES * sizeof(float)),
#if AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_CMVN
        .cmvn = (float *)malloc(AUDIO_LOGMEL_PLANE_VALUES * sizeof(float)),
#else
        .cmvn = NULL,
#endif
        .normalized_input = (float *)malloc(
            AUDIO_LOGMEL_MODEL_INPUT_VALUES * sizeof(float)),
    };
    if (pcm == NULL || model_input == NULL || workspace == NULL ||
        debug.dc_removed_audio == NULL || debug.power_spectrogram == NULL ||
        debug.mel_power == NULL || debug.log_mel == NULL || debug.delta == NULL ||
#if AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_CMVN
        debug.cmvn == NULL ||
#endif
        debug.normalized_input == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 3;
    }
    if (read_exact(argv[1], pcm, sizeof(int16_t), AUDIO_LOGMEL_INPUT_SAMPLES) != 0) {
        return 4;
    }

    int result = audio_logmel_extract_normalized_pcm16(
        pcm,
        AUDIO_LOGMEL_INPUT_SAMPLES,
        model_input,
        workspace,
        audio_logmel_workspace_size(),
        &debug);
    if (result != AUDIO_LOGMEL_OK) {
        fprintf(stderr, "feature extraction failed: %d\n", result);
        return 5;
    }

    int write_result = 0;
    write_result |= write_floats(
        argv[2], "dc_removed.bin", debug.dc_removed_audio, AUDIO_LOGMEL_INPUT_SAMPLES);
    write_result |= write_floats(
        argv[2], "power.bin", debug.power_spectrogram,
        AUDIO_LOGMEL_FRAME_COUNT * AUDIO_LOGMEL_FFT_BINS);
    write_result |= write_floats(
        argv[2], "mel_power.bin", debug.mel_power, AUDIO_LOGMEL_PLANE_VALUES);
    write_result |= write_floats(
        argv[2], "log_mel.bin", debug.log_mel, AUDIO_LOGMEL_PLANE_VALUES);
    write_result |= write_floats(
        argv[2], "delta.bin", debug.delta, AUDIO_LOGMEL_PLANE_VALUES);
#if AUDIO_MODEL_FEATURE_MODE == AUDIO_MODEL_FEATURE_CMVN
    write_result |= write_floats(
        argv[2], "cmvn.bin", debug.cmvn, AUDIO_LOGMEL_PLANE_VALUES);
#endif
    write_result |= write_floats(
        argv[2], "normalized.bin", debug.normalized_input,
        AUDIO_LOGMEL_MODEL_INPUT_VALUES);
    if (write_result != 0) {
        fprintf(stderr, "failed to write one or more stage files\n");
        return 6;
    }

    free(debug.normalized_input);
    free(debug.cmvn);
    free(debug.delta);
    free(debug.log_mel);
    free(debug.mel_power);
    free(debug.power_spectrogram);
    free(debug.dc_removed_audio);
    free(workspace);
    free(model_input);
    free(pcm);
    return 0;
}
