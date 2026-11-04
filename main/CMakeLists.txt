# Float32 firmware source set.
#
# Exactly one application source containing app_main() may be registered:
#   - app_audio_infer.c: standard I2S microphone (current selection)
#   - app_audio_infer_pdm.c: PDM microphone
#
# model_runner.cc, model_data.cc, logmel_features.c, and preprocess_config.h
# form one deployment contract. Regenerate the model/config together and do
# not mix these float32 files with their *_int8 counterparts.

idf_component_register(SRCS "model_data_int8.cc" "logmel_features_int8.c" "logmel_features.c" "app_audio_infer.c" "model_data.cc" "model_runner.cc" "logmel_features.c" "memory_monitor.c"
#idf_component_register(SRCS "pdm_recorder_dual_task.c" 
                    PRIV_REQUIRES app_update esp_psram esp_timer driver fatfs sdmmc spi_flash espressif__esp-tflite-micro
                    INCLUDE_DIRS ".")
