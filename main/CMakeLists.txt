# Firmware source set: continuous capture (Core 0) + full-INT8 inference (Core 1).
#
# Exactly one application source containing app_main() may be registered:
#   - app_audio_stream_int8.c: records from boot, infers the previous 2 s window
#     on the other core (current selection). I2S or PDM mic via -DAUDIO_MIC
#     (see the top-level CMakeLists.txt).
#   - app_audio_infer_int8.c / app_audio_infer_pdm_int8.c: button-triggered int8
#   - app_audio_infer.c / app_audio_infer_pdm.c: button-triggered float32; these
#     need model_runner.cc + model_data.cc + logmel_features.c instead
#
# model_runner_int8.cc, model_data_int8.cc, logmel_features_int8.c, and
# preprocess_config_int8.h form one deployment contract; regenerate them together
# with tools/convert_logmel_tflite_to_c_int8.py.

idf_component_register(
    SRCS
        "app_audio_stream_int8.c"
        "model_runner_int8.cc"
        "model_data_int8.cc"
        "logmel_features_int8.c"
        "memory_monitor.c"
    PRIV_REQUIRES
        app_update
        esp_psram
        esp_timer
        driver
        fatfs
        sdmmc
        spi_flash
        vfs
        espressif__esp-tflite-micro
    INCLUDE_DIRS "."
)

# AUDIO_MIC is undefined while IDF scans component requirements; default to I2S.
if(AUDIO_MIC STREQUAL "pdm")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE STREAM_MIC_PDM=1)
elseif(DEFINED AUDIO_MIC AND NOT AUDIO_MIC STREQUAL "i2s")
    message(FATAL_ERROR "Unknown AUDIO_MIC=${AUDIO_MIC} (expected i2s or pdm)")
endif()
