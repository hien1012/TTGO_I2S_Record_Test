#pragma once

/**
 * @file model_data.h
 * @brief Linkage contract for the generated float32 TFLite flatbuffer.
 *
 * Generated model bytes reside in Flash as a const, 16-byte-aligned array.
 * Replace this header and model_data.cc together; never edit the byte array.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Embedded TFLite flatbuffer and its exact byte length. */
extern const unsigned char g_audio_model_tflite[];
extern const size_t g_audio_model_tflite_len;

#ifdef __cplusplus
}
#endif
