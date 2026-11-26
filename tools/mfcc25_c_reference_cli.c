/**
 * @file mfcc25_c_reference_cli.c
 * @brief Host executable exposing firmware MFCC25 stages for parity testing.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mfcc25_features.h"
#include "mfcc25_model_config.h"

static int read_pcm16(const char *path, int16_t *pcm)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return -1;
    }
    size_t count = fread(pcm, sizeof(*pcm), MFCC25_INPUT_SAMPLES, file);
    int extra = fgetc(file);
    fclose(file);
    return count == MFCC25_INPUT_SAMPLES && extra == EOF ? 0 : -1;
}

static int write_binary(
    const char *directory,
    const char *name,
    const void *data,
    size_t element_size,
    size_t count)
{
    char path[512];
    int length = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (length <= 0 || (size_t)length >= sizeof(path)) {
        return -1;
    }
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return -1;
    }
    size_t written = fwrite(data, element_size, count, file);
    fclose(file);
    return written == count ? 0 : -1;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s input_pcm16.bin output_directory\n", argv[0]);
        return 2;
    }

    int16_t *pcm = (int16_t *)malloc(MFCC25_INPUT_SAMPLES * sizeof(int16_t));
    void *workspace = malloc(mfcc25_workspace_size());
    float *centered = (float *)malloc(MFCC25_INPUT_SAMPLES * sizeof(float));
    float *energy = (float *)malloc(MFCC25_FRAME_COUNT * sizeof(float));
    float *mfcc = (float *)malloc(
        MFCC25_N_MFCC * MFCC25_FRAME_COUNT * sizeof(float));
    float *delta = (float *)malloc(
        MFCC25_N_MFCC * MFCC25_FRAME_COUNT * sizeof(float));
    if (pcm == NULL || workspace == NULL || centered == NULL || energy == NULL ||
        mfcc == NULL || delta == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 3;
    }
    if (read_pcm16(argv[1], pcm) != 0) {
        fprintf(stderr, "input must contain exactly %d PCM16 samples\n", MFCC25_INPUT_SAMPLES);
        return 4;
    }

    float raw[MFCC25_FEATURE_DIM];
    float normalized[MFCC25_FEATURE_DIM];
    int8_t quantized[MFCC25_FEATURE_DIM];
    mfcc25_debug_buffers_t debug = {
        .centered_audio = centered,
        .frame_energy_db = energy,
        .mfcc_frames = mfcc,
        .delta_frames = delta,
    };
    int result = mfcc25_extract_raw_pcm16(
        pcm,
        MFCC25_INPUT_SAMPLES,
        raw,
        workspace,
        mfcc25_workspace_size(),
        &debug);
    if (result != MFCC25_OK) {
        fprintf(stderr, "MFCC25 extraction failed: %d\n", result);
        return 5;
    }
    mfcc25_normalize(raw, normalized);
    mfcc25_quantize_input(normalized, quantized);

    int write_failed =
        write_binary(argv[2], "centered.bin", centered, sizeof(float), MFCC25_INPUT_SAMPLES) ||
        write_binary(argv[2], "energy.bin", energy, sizeof(float), MFCC25_FRAME_COUNT) ||
        write_binary(argv[2], "mfcc.bin", mfcc, sizeof(float), MFCC25_N_MFCC * MFCC25_FRAME_COUNT) ||
        write_binary(argv[2], "delta.bin", delta, sizeof(float), MFCC25_N_MFCC * MFCC25_FRAME_COUNT) ||
        write_binary(argv[2], "raw_features.bin", raw, sizeof(float), MFCC25_FEATURE_DIM) ||
        write_binary(argv[2], "normalized.bin", normalized, sizeof(float), MFCC25_FEATURE_DIM) ||
        write_binary(argv[2], "quantized.bin", quantized, sizeof(int8_t), MFCC25_FEATURE_DIM);
    if (write_failed) {
        fprintf(stderr, "failed to write one or more stage files\n");
        return 6;
    }

    free(delta);
    free(mfcc);
    free(energy);
    free(centered);
    free(workspace);
    free(pcm);
    return 0;
}

