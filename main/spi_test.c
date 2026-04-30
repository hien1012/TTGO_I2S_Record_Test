#include <stdio.h>
#include <string.h>
#include <sys/unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include "esp_err.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#include "driver/gpio.h"
#include "hal/gpio_hal.h"

#define SD_MISO_PIN     GPIO_NUM_2
#define SD_MOSI_PIN     GPIO_NUM_15
#define SD_CLK_PIN      GPIO_NUM_14
#define SD_CS_PIN       GPIO_NUM_13
// 定義系統 Log 標籤
static const char *TAG = "SD_MMC_test";

// 定義掛載點與腳位 (針對部分支援 GPIO Matrix 的 ESP32 晶片/IDF 版本)
#define MOUNT_POINT "/sdcard"

void app_main(void)
{   
    esp_err_t ret;

    // ==========================================
    // 1. 設定 VFS FAT 掛載選項
    // ==========================================
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        // 專業實務：不要在掛載失敗時自動格式化 SD 卡，避免銷毀使用者重要資料
        .format_if_mount_failed = false, 
        .max_files = 5,                      // 同時開啟的最大檔案數量
        .allocation_unit_size = 16 * 1024    // 配置單位大小
    };
    sdmmc_card_t *card;

    ESP_LOGI(TAG, "Initializing SD card via SDMMC peripheral (1-Bit Mode)");

    // ==========================================
    // 2. 初始化 SDMMC Host 驅動
    // ==========================================
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = SPI2_HOST;

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = SD_MOSI_PIN,
        .miso_io_num = SD_MISO_PIN,
        .sclk_io_num = SD_CLK_PIN,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4096,
    };
    //ESP_RETURN_ON_ERROR(spi_bus_initialize(host.slot, &bus_cfg, SPI_DMA_CH_AUTO), TAG, "spi_bus_initialize failed");
    spi_bus_initialize(host.slot, &bus_cfg, SPI_DMA_CH_AUTO);
    // ==========================================
    // 3. 初始化 SPI Slot 配置
    // ==========================================
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = SD_CS_PIN;
    slot_config.host_id = host.slot;


    // ==========================================
    // 4. 掛載檔案系統
    // ==========================================
    ESP_LOGI(TAG, "Starting SDMMC mounting...");
    
    ret = esp_vfs_fat_sdspi_mount(MOUNT_POINT, &host, &slot_config, &mount_config, &card);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount filesystem. SD card might be corrupted or unformatted.");
        } else {
            ESP_LOGE(TAG, "SDMMC init failed with error code: 0x%X (%s)", ret, esp_err_to_name(ret));
        }
        return;
    }
    
    ESP_LOGI(TAG, "Filesystem mounted successfully");
    
    // 印出 SD 卡的底層資訊 (容量、速度、廠商等)
    sdmmc_card_print_info(stdout, card);

    // ==========================================
    // 5. 檔案操作測試 (Write & Read)
    // ==========================================
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "═══ 檔案操作測試 ═══");
    
    // 第一步：檢查掛載點是否存在
    struct stat st;
    if (stat(MOUNT_POINT, &st) == 0) {
        ESP_LOGI(TAG, " Mount point %s exists", MOUNT_POINT);
    } else {
        ESP_LOGE(TAG, " Mount point %s does not exist! Error: %s", MOUNT_POINT, strerror(errno));
        esp_vfs_fat_sdcard_unmount(MOUNT_POINT, card);
        return;
    }
    
    // 【修改】使用 8.3 FAT32 格式文件名而不是長文件名
    // 原因：某些 FAT32 實現對長文件名支持有限
    const char *file_path = MOUNT_POINT "/TEST.TXT";  
    ESP_LOGI(TAG, "Opening file %s for write ", file_path);
    
    FILE *f = fopen(file_path, "w");
    if (f == NULL) { /* 如果file創建有問題，就進行診斷 */
        ESP_LOGE(TAG, "Failed to open file for writing!");
        ESP_LOGE(TAG, "Error code: %d (%s)", errno, strerror(errno));
        
        // 診斷信息
        ESP_LOGE(TAG, "");
        ESP_LOGE(TAG, "【檔案系統診斷】");
        ESP_LOGE(TAG, "1. 嘗試列表 %s 中的檔案...", MOUNT_POINT);
        
        DIR *dir = opendir(MOUNT_POINT);
        if (dir == NULL) {
            ESP_LOGE(TAG, "   Cannot open directory: %s", strerror(errno));
        } else {
            struct dirent *entry;
            int file_count = 0;
            while ((entry = readdir(dir)) != NULL) {
                file_count++;
                ESP_LOGI(TAG, "   - %s", entry->d_name);
            }
            closedir(dir);
            
            if (file_count == 0) {
                ESP_LOGW(TAG, "   (SD 卡為空)");
            }
        }
        
        // 嘗試用各種文件名格式進行測試
        ESP_LOGE(TAG, "2. 嘗試不同的文件名格式...");
        
        const char *test_names[] = {
            MOUNT_POINT "/T.TXT",           // 最短的 8.3 格式
            MOUNT_POINT "/TEST.TXT",       // 標準 8.3 格式
            MOUNT_POINT "/F.DAT",          // 不同的副檔名
            MOUNT_POINT "/test.txt",       // 小寫
        };
        
        for (int i = 0; i < sizeof(test_names) / sizeof(test_names[0]); i++) {
            FILE *test_f = fopen(test_names[i], "w");
            if (test_f != NULL) {
                fprintf(test_f, "Success with: %s\n", test_names[i]);
                fclose(test_f);
                ESP_LOGI(TAG, "成功：%s", test_names[i]);
                
                // 現在嘗試用這個文件名重新打開
                f = fopen(test_names[i], "r+");  // 改為 r+ 讀寫模式
                if (f != NULL) {
                    ESP_LOGI(TAG, "成功打開並讀寫該文件！");
                    fclose(f);
                    break;
                }
            } else {
                ESP_LOGD(TAG, "   ✗ 失敗：%s (errno=%d)", test_names[i], errno);
            }
        }
        
        if (f == NULL) {
            ESP_LOGE(TAG, "");
            ESP_LOGE(TAG, "【可能的原因】");
            ESP_LOGE(TAG, "• SD 卡 FAT32 文件系統損壞");
            ESP_LOGE(TAG, "• 長文件名 (LFN) 支援問題");
            ESP_LOGE(TAG, "• 文件系統讀寫保護啟用");
            ESP_LOGE(TAG, "建議：在電腦上重新格式化 SD 卡為 FAT32");
            esp_vfs_fat_sdcard_unmount(MOUNT_POINT, card);
            return;
        }
    } else {
        ESP_LOGI(TAG, "成功打開文件進行寫入");
    }
    
    // 寫入資料, 同時輸出在terminal上
    fprintf(f, "SD/SPI success");
    fclose(f);
    ESP_LOGI(TAG, "File written successfully");
    
    // 讀取檔案
    ESP_LOGI(TAG, "Reading file back to verify...");
    FILE *f_read = fopen(file_path, "r");
    if (f_read != NULL) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), f_read) != NULL) {
            ESP_LOGI(TAG, "File content: %s", buffer);
        }
        fclose(f_read);
    }
    ESP_LOGI(TAG, "═══ 測試完成 ═══");

    // ==========================================
    // 6. 安全卸載 (Unmount)
    // ==========================================
    ret = esp_vfs_fat_sdcard_unmount(MOUNT_POINT, card);
    if(ret == ESP_OK)
        ESP_LOGI(TAG, " Card unmounted safely");

        
    //while(1);
}