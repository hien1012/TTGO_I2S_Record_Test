/*
 * Batch host runner for the legacy firmware MFCC frontend (main/mfcc.c).
 *
 * mfcc.c is compiled unmodified; ESP-IDF headers come from tools/mfcc_host_stubs.
 * Input:  count x 88200 little-endian int16 samples (2 s at 44.1 kHz).
 * Output: count x 26 float32 values (13 MFCC means followed by 13 stds).
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MFCC_INPUT_SAMPLES 88200
#define MFCC_OUTPUT_VALUES 26
#define MFCC_FFT_SIZE 512
#define MFCC_SAMPLE_RATE 44100
#define MFCC_NUM_FILTERS 26

/* Definitions live in main/mfcc.c, which has no usable header. */
float *create_mel_filterbank(int num_filters, int fft_size, int sample_rate);
float *process_int16_buffer(float *filterbank, int16_t *buffer);

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

    float *filterbank = create_mel_filterbank(MFCC_NUM_FILTERS, MFCC_FFT_SIZE, MFCC_SAMPLE_RATE);
    if (filterbank == NULL) {
        fprintf(stderr, "filterbank allocation failed\n");
        return 4;
    }

    for (size_t clip = 0; clip < count; ++clip) {
        /* process_int16_buffer() frees its input, so every clip gets a fresh buffer. */
        int16_t *pcm = (int16_t *)malloc(MFCC_INPUT_SAMPLES * sizeof(int16_t));
        if (pcm == NULL) {
            fprintf(stderr, "pcm allocation failed at clip %u\n", (unsigned)clip);
            return 5;
        }
        if (fread(pcm, sizeof(int16_t), MFCC_INPUT_SAMPLES, in) != MFCC_INPUT_SAMPLES) {
            fprintf(stderr, "short read at clip %u\n", (unsigned)clip);
            return 6;
        }
        float *features = process_int16_buffer(filterbank, pcm);
        if (features == NULL) {
            fprintf(stderr, "mfcc extraction failed at clip %u\n", (unsigned)clip);
            return 7;
        }
        if (fwrite(features, sizeof(float), MFCC_OUTPUT_VALUES, out) != MFCC_OUTPUT_VALUES) {
            fprintf(stderr, "short write at clip %u\n", (unsigned)clip);
            return 8;
        }
        free(features);
    }

    free(filterbank);
    fclose(in);
    fclose(out);
    return 0;
}
