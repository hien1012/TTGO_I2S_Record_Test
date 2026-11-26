/*
 * Host batch runner for Xu's MFCC front end (mfcc_function-2.c), used only by
 * tools/check_xu_mfcc_parity.py to compare the NumPy port against the C code.
 *
 * Build:  gcc -O2 -std=gnu99 -DXU_MFCC_SOURCE="\"path/to/mfcc_function-2.c\"" xu_mfcc_c_cli.c -lm
 * Input:  count x samples float32 values on the int16 scale (as the notebook passes them).
 * Output: count x 26 float32 values (13 MFCC means followed by 13 stds).
 */
#include XU_MFCC_SOURCE

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s input_f32.bin output_f32.bin clip_count samples_per_clip\n", argv[0]);
        return 2;
    }
    size_t count = strtoul(argv[3], NULL, 10);
    size_t samples = strtoul(argv[4], NULL, 10);
    FILE *in = fopen(argv[1], "rb");
    FILE *out = fopen(argv[2], "wb");
    if (in == NULL || out == NULL) {
        fprintf(stderr, "cannot open input/output\n");
        return 3;
    }
    float *clip = (float *)malloc(samples * sizeof(float));
    if (clip == NULL) {
        return 4;
    }
    for (size_t i = 0; i < count; ++i) {
        if (fread(clip, sizeof(float), samples, in) != samples) {
            fprintf(stderr, "short read at clip %u\n", (unsigned)i);
            return 5;
        }
        float *features = extract_mfcc_from_array(clip, (uint32_t)samples);
        if (features == NULL || fwrite(features, sizeof(float), 26, out) != 26) {
            fprintf(stderr, "extraction/write failed at clip %u\n", (unsigned)i);
            return 6;
        }
        free_mfcc(features);
    }
    free(clip);
    fclose(in);
    fclose(out);
    return 0;
}
