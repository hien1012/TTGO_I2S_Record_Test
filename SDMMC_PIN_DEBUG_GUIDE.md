# SDMMC 脚位调试指南

## 📌 方法一：运行时输出（推荐）

现在已在 `sdmmc.c` 中添加了 `print_sdmmc_pins_info()` 函数，会在以下时机打印脚位信息：

1. **GPIO 配置后** - 初始化前检查
2. **挂载成功后** - 初始化后检查

**运行输出示例：**
```
I (359) SD_MMC_test: ╔═══════════════════════════════════════════════════════╗
I (369) SD_MMC_test: ║         SDMMC 脚位配置信息                           ║
I (379) SD_MMC_test: ╚═══════════════════════════════════════════════════════╝
I (389) SD_MMC_test: 【SDMMC Slot 1 標準脚位 (1-Bit Mode)】
I (399) SD_MMC_test:   GPIO 14 (CLK): Level=1
I (409) SD_MMC_test:   GPIO 15 (CMD): Level=1
I (419) SD_MMC_test:   GPIO  2 (DAT0): Level=1
I (429) SD_MMC_test: 【配置信息】
I (439) SD_MMC_test:   Bus Width: 1-bit (only DAT0 used)
I (449) SD_MMC_test:   Slot: 1 (default)
I (459) SD_MMC_test:   内部上拉: 启用
```

**GPIO Level 说明：**
- `Level=1` ✅ 正常（上拉至高电平）
- `Level=0` ❌ 异常（可能接线松动或 SD 卡未就绪）

---

## 📂 方法二：从 ESP-IDF 驱动源码查看

### 标准 SDMMC Slot 1 脚位定义位置：

**文件位置：** 
```
$IDF_PATH/components/driver/gpio/gpio_ioctl.c
$IDF_PATH/components/esp_driver_sdmmc/src/sdmmc_init.c
$IDF_PATH/components/hal/esp32/include/hal/sdio_ll.h
```

### 快速查找方法：

```bash
# 1. 查找 SDMMC Slot 配置的默认脚位
grep -r "SDMMC_SLOT_CONFIG_DEFAULT" $IDF_PATH/components/

# 2. 查找 GPIO Matrix 配置
grep -r "SOC_SDMMC_USE_GPIO_MATRIX" $IDF_PATH/components/

# 3. 查看特定芯片的 SDMMC 脚位
grep -r "SDIO_LL" $IDF_PATH/components/hal/esp32/
```

### 对于标准 ESP32，Slot 1 的固定脚位：

**在 `esp-idf/components/hal/esp32/include/hal/sdio_ll.h` 中：**

```c
// SDMMC Slot 1 使用以下脚位（固定，不可改）
#define SDMMC_SLOT1_CLK_GPIO    14   // CLK
#define SDMMC_SLOT1_CMD_GPIO    15   // CMD
#define SDMMC_SLOT1_D0_GPIO      2   // DAT0
#define SDMMC_SLOT1_D1_GPIO      4   // DAT1 (4-bit mode)
#define SDMMC_SLOT1_D2_GPIO     12   // DAT2 (4-bit mode)
#define SDMMC_SLOT1_D3_GPIO     13   // DAT3 (4-bit mode)
```

---

## 🔍 方法三：通过 menuconfig 查看配置

```bash
idf.py menuconfig
```

导航到：
```
Component config → SD/MMC → SDMMC Controller
  ├─ Enable 1-line (DAT0) mode
  ├─ Slot 1 (SDMMC) pins
  └─ Maximum SDMMC frequency
```

---

## 🧪 进阶调试：检查 GPIO 中断/事件

如果怀疑脚位有问题，可以添加此代码：

```c
static void check_gpio_interrupts(void)
{
    ESP_LOGI(TAG, "Checking GPIO interrupt status...");
    
    uint32_t gpio_intr_status = READ_PERI_REG(GPIO_STATUS_REG);
    ESP_LOGI(TAG, "GPIO interrupt status: 0x%08X", gpio_intr_status);
    
    if (gpio_intr_status & BIT(SD_CLK_PIN)) {
        ESP_LOGW(TAG, "GPIO 14 (CLK) has pending interrupt!");
    }
    if (gpio_intr_status & BIT(SD_CMD_PIN)) {
        ESP_LOGW(TAG, "GPIO 15 (CMD) has pending interrupt!");
    }
    if (gpio_intr_status & BIT(SD_D0_PIN)) {
        ESP_LOGW(TAG, "GPIO 2 (DAT0) has pending interrupt!");
    }
}
```

---

## 📋 常见问题排查

| 症状 | 原因 | 解决方案 |
|------|------|--------|
| Level=0 on all pins | 无供电/接线错误 | 检查 3.3V 供电和 GND 连接 |
| Level=0 on CLK only | CLK 脚位接线松动 | 重新焊接 GPIO 14 连接 |
| GPIO 配置成功但仍失败 | 上拉电阻不足 | 添加外部 10kΩ 上拉电阻 |
| 驱动加载但无响应 | 时钟频率过高 | 降低至 400 kHz 测试 |

---

## 🎯 快速验证脚位流程

```bash
# 1. 编译和烧录
idf.py build
idf.py flash

# 2. 监视输出
idf.py monitor

# 3. 查看打印的脚位信息
# 应该看到 GPIO 14, 15, 2 都是 Level=1

# 4. 如果失败，尝试 400 kHz (已在代码中启用)

# 5. 检查硬件
# - 万用表测量 GPIO 14, 15, 2 对 GND 的电压（应该 ~3.3V）
# - 检查焊点是否冷焊或断开
```

---

## 📖 参考资源

- ESP-IDF SDMMC 官方文档：https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/sdmmc_host.html
- HAL GPIO 层代码：`$IDF_PATH/components/hal/`
- 驱动源码：`$IDF_PATH/components/esp_driver_sdmmc/`

