idf_component_register(SRCS "i2s_recorder_dual_task.c" 
                    PRIV_REQUIRES nvs_flash esp_psram fatfs esp_timer driver sdmmc
                    INCLUDE_DIRS "$ENV{IDF_PATH}/examples/peripherals/i2s/common")
