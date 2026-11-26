# TTGO I2S Record Test：ESP32 端無人機／直升機聲音辨識

這個專案在 ESP32 上用 MEMS 麥克風（I2S 或 PDM）連續收音，每 2 秒算一次 Log-Mel + CMVN 特徵，再用 **full-INT8 的小型 CNN**（TensorFlow Lite Micro）把聲音分成三類：

| 類別 index | 類別名稱 |
|:---:|:---|
| 0 | `background` |
| 1 | `drone` |
| 2 | `helicopter` |

專案包含兩部分：

1. **韌體（`main/`）**：ESP-IDF 專案，負責收音、前處理、推論、把 WAV 存到 SD 卡。
2. **PC 端 Python 工具鏈（根目錄與 `tools/`）**：從 WAV 資料集訓練模型、量化成 INT8 TFLite、轉成 C 陣列嵌入韌體，並檢查 Python 和 C 的前處理是否逐值一致。

> 支援目標：**ESP32（需 PSRAM）**。ESP-IDF 版本 **5.1.2**（`idf_component.yml` 要求 `>=5.1.0`）。

---

## 目錄

- [系統架構](#系統架構)
- [目錄結構](#目錄結構)
- [硬體與接線](#硬體與接線)
- [韌體：建置與燒錄](#韌體建置與燒錄)
- [前處理規格（部署契約）](#前處理規格部署契約)
- [模型架構](#模型架構)
- [Python 環境](#python-環境)
- [資料集準備](#資料集準備)
- [訓練模型](#訓練模型)
- [匯出模型到韌體](#匯出模型到韌體)
- [評估與 Python／C 一致性驗證](#評估與-pythonc-一致性驗證)
- [比較實驗腳本](#比較實驗腳本)
- [注意事項與常見問題](#注意事項與常見問題)

---

## 系統架構

```
 ┌──────────────────────────── PC（Python） ────────────────────────────┐
 │                                                                      │
 │  WAV 資料集 ──► 依原始檔切 train/val ──► 切成 2 s 片段 ──► Log-Mel     │
 │                                   │                                  │
 │                                   └─► 只對 train 做資料增強           │
 │                                                                      │
 │  CMVN ──► 訓練 Keras CNN ──► BN 再校正 ──► float32 / full-INT8 TFLite │
 │                                                                      │
 │  tools/convert_logmel_tflite_to_c_int8.py                            │
 │      └─► main/model_data_int8.cc + main/preprocess_config_int8.h     │
 └──────────────────────────────────────────────────────────────────────┘
                                   │  idf.py build flash
                                   ▼
 ┌──────────────────────────── ESP32 韌體 ──────────────────────────────┐
 │  Core 0  mic_capture  (prio 7)  I2S/PDM DMA → PCM16 ×30 → 2 s 視窗    │
 │  Core 0  logmel_task  (prio 4)  Log-Mel + CMVN (float32) → int8 量化  │
 │  Core 1  cnn_task     (prio 3)  INT8 CNN 推論，輸出三類機率           │
 │  Core 1  audio_writer (prio 2)  依條件把視窗寫成 WAV + CSV 到 SD 卡    │
 └──────────────────────────────────────────────────────────────────────┘
```

錄第 N 個視窗時，另一顆核心同時推論第 N-1 個視窗，所以收音不會因為推論或 SD 寫入而中斷。

---

## 目錄結構

### 韌體 `main/`

| 檔案 | 說明 |
|:---|:---|
| `app_audio_stream_int8.c` | **目前使用的主程式**。開機後就持續收音，雙核心管線推論（見上圖）。 |
| `app_audio_infer_int8.c` / `app_audio_infer_pdm_int8.c` | 按鈕觸發版本（I2S / PDM），按一次錄 2 秒 → 存 WAV → INT8 推論。 |
| `app_audio_infer.c` / `app_audio_infer_pdm.c` | 按鈕觸發的 float32 模型版本（舊版）。 |
| `logmel_features_int8.c/.h` | Log-Mel + CMVN 前處理（float32 運算，最後量化成 int8），不使用 heap。 |
| `model_runner_int8.cc/.h` | TFLite Micro 解譯器的封裝；tensor arena 256 KB，配置在 PSRAM。 |
| `model_data_int8.cc/.h` | 嵌入的 INT8 TFLite flatbuffer（**自動產生**）。 |
| `preprocess_config_int8.h` | 特徵尺寸、Mel 頻帶邊界、量化 scale/zero-point、類別名稱（**自動產生**）。 |
| `logmel_conversion_summary_int8.json` | 目前嵌入的模型路徑、SHA-256 和量化參數紀錄。 |
| `memory_monitor.c/.h` | 在序列埠印出 heap／PSRAM／stack 使用量。 |
| `logmel_features.c`、`model_*.cc`、`preprocess_config.h` | float32 版本對應的檔案。 |
| `Kconfig.projbuild` | 原始範例留下的選單設定；目前的主程式**沒有使用**這些選項，腳位直接寫在 `.c` 檔裡。 |

### PC 端 Python

| 檔案 | 說明 |
|:---|:---|
| `train_cnn_from_wav_logmel_cmvn.py` | **主要的訓練入口**。設定好 CMVN 版本的預設值後呼叫下面那支。 |
| `train_cnn_from_wav_logmel.py` | 通用的 Log-Mel CNN 訓練流程（切資料、增強、訓練、匯出、報告）。 |
| `train_cnn_from_wav_logmel_cmvn_sliding.py` | 實驗用：重疊滑動視窗＋裝置／戶外域增強的訓練器。 |
| `log_mel_features.py` | Log-Mel 特徵計算；和 C 版 `logmel_features*.c` 一一對應。 |
| `logmel_cnn.py` | 模型結構、CMVN 等衍生通道、BN 再校正、TFLite 匯出。 |
| `audio_dataset.py` | 讀 WAV、從檔名推標籤、Dataset 結構、輸出 CSV 報告。 |
| `long_wav_segment_dataset.py` | 先依「原始錄音檔」分 train/val，再切固定長度片段，避免資料外洩。 |
| `sliding_window_logmel_dataset.py` | 滑動視窗版的資料集建構（同樣避免外洩）。 |
| `audio_data_augmentation.py` | 訓練用波形增強：DC 移除、EQ、時移、增益、高斯雜訊、背景混音。 |
| `device_domain_augmentation.py` | 裝置／戶外域增強：IR、EQ、頻譜傾斜、感測器雜訊、殘響等。 |
| `training_reports.py` | 混淆矩陣、分類報告、錯誤樣本匯出、訓練曲線。 |
| `quantize_logmel_cnn_int8.py` | 把現有的 Keras 模型轉成 full-INT8 TFLite。 |
| `evaluate_trained_audio_folders.py` | 在有標籤的 WAV 資料夾上評估模型，分麥克風統計。 |
| `evaluate_logmel_tflite_int8.py` | 評估 INT8 TFLite，並可和 float 模型比對。 |
| `evaluate_logmel_gain_sensitivity.py` | 測試預測結果對輸入增益的敏感度。 |
| `compare_logmel_python_c.py` / `compare_logmel_python_c_int8.py` | 用 gcc 編譯韌體的 C 前處理，逐階段和 Python 比對。 |
| `run_c_logmel_int8_inference.py` | 在 PC 上用韌體的 C 前處理 + 嵌入的 INT8 模型跑 WAV（可逐位元重現裝置結果）。 |
| `compare_infer_node_debug_dump.py` | 比對 infer_node 裝置的 debug dump 和 PC 參考值（逐層）。 |
| `run_mic_loo_comparison.py` | 跨麥克風實驗：Xu MFCC-DNN 和 Log-Mel CMVN CNN 的比較。 |
| `run_ltcmn_mic_loo_comparison.py` | 跨麥克風實驗：per-clip CMVN 和長期 CMN（LTCMN）的比較。 |
| `run_xu_sample_rate_check.py` | 檢查 Xu 的 MFCC-DNN 從 44.1 kHz 改成 16 kHz 是否影響準確率。 |
| `train_xu_mfcc_dnn.py` / `xu_mfcc_features.py` | Xu 的 MFCC + DNN 基準模型（NumPy 移植）。 |
| `apply_wav_gain_folder.py`、`clip_wav.py`、`invert_negative_dc_wavs.py` | WAV 前處理小工具（加增益、剪頭尾、修正反相）。 |
| `tests/test_sliding_window_voting.py` | 滑動視窗投票邏輯的測試。 |

### `tools/`

| 檔案 | 說明 |
|:---|:---|
| `convert_logmel_tflite_to_c_int8.py` | INT8 TFLite + `preprocess_config.json` → `main/model_data_int8.cc`、`main/preprocess_config_int8.h`。 |
| `convert_logmel_tflite_to_c.py` | float32 版本的轉換器。 |
| `logmel_mel_band.py` | 兩支轉換器共用的 Mel 頻帶邊界邏輯。 |
| `logmel_c_reference_cli*.c`、`logmel_c_batch_cli*.c` | PC 端呼叫韌體 C 前處理的小程式（由比對腳本自動編譯）。 |
| `receive_infer_node_dump.py` | 從序列埠（或存好的 log）接收裝置的 debug dump 檔案。 |
| `check_xu_mfcc_parity.py`、`xu_mfcc_c_cli.c`、`mfcc*_cli.c` | MFCC 前處理的 Python／C 一致性檢查。 |
| `mix.py`、`move.py`、`split_wav.py`、`tiltle_edit.py` | 資料整理小工具（混合麥克風資料、切 WAV、批次改檔名）。 |

### 產出資料夾（已被 `.gitignore` 排除）

`raw_audio/`（原始錄音）、`cnn_output/`（訓練結果）、`build*/`、`evaluation*/`、`infer_node*/` 等。

---

## 硬體與接線

- ESP32 開發板（TTGO，**需要 PSRAM**：四個 64 KB 音訊視窗、特徵緩衝區和 tensor arena 都放在 PSRAM）
- 一支麥克風，二選一：
  - 標準 I2S 麥克風（例如 INMP441、ICS-43434）
  - PDM 麥克風
- microSD 卡（FAT 格式；韌體不會自動格式化）

### 麥克風

| 標準 I2S 麥克風 | GPIO |
|:---|:---:|
| BCLK（SCK） | 26 |
| WS（LRCL） | 25 |
| DIN（SD） | 33 |
| L/R | 接 GND（讀左聲道） |

| PDM 麥克風 | GPIO |
|:---|:---:|
| CLK | 4 |
| DATA | 5 |

### SD 卡

使用 **SDMMC 1-bit 模式**（20 MHz，啟用內部上拉），腳位是 ESP32 SDMMC Slot 1 的固定腳位：

| SD 卡 | GPIO |
|:---|:---:|
| CLK | 14 |
| CMD | 15 |
| D0 | 2 |

### 除錯／其他腳位

| 用途 | GPIO | 說明 |
|:---|:---:|:---|
| `DBG_CNN_PIN` | 12 | CNN 推論期間拉高（串流版） |
| `DBG_REC_PIN` | 12 | 錄音期間拉高（按鈕版） |
| `DBG_SD_PIN` | 18 | 寫 SD 卡期間拉高 |
| `BTN_PIN` | 0 | 按鈕版的觸發按鈕（BOOT 鍵） |

可以用邏輯分析儀或示波器量這些腳位，看推論和寫入花了多少時間。

> 要改腳位，請直接修改對應 `app_*.c` 開頭的 `#define`。

---

## 韌體：建置與燒錄

### 1. 準備 ESP-IDF

安裝 ESP-IDF 5.1.x，並在 ESP-IDF 終端機（或已執行 `export.ps1` / `export.sh` 的環境）中操作。專案依賴的元件由 Component Manager 管理：

- `espressif/esp-tflite-micro` 1.3.7
- `espressif/esp-nn` 1.2.3

`main/idf_component.yml` 用 `override_path` 指向 `managed_components/` 底下的本地副本，所以第一次建置前，請確認 `managed_components/espressif__esp-tflite-micro` 和 `managed_components/espressif__esp-nn` 這兩個資料夾存在。

### 2. 選擇麥克風並建置

串流主程式用 CMake 變數 `AUDIO_MIC` 選麥克風，這個值會快取在各自的 build 資料夾裡，建議兩種麥克風分開建置：

```powershell
# 標準 I2S 麥克風（預設）
idf.py -B build_i2s build
idf.py -B build_i2s -p COM3 flash monitor

# PDM 麥克風
idf.py -B build_pdm -DAUDIO_MIC=pdm build
idf.py -B build_pdm -p COM3 flash monitor
```

（離開 monitor：`Ctrl-]`）

### 3. 切換主程式

`main/CMakeLists.txt` 的 `SRCS` 裡**只能放一個含有 `app_main()` 的檔案**。預設是 `app_audio_stream_int8.c`；如果要改用按鈕觸發版：

| 想要的模式 | 放進 `SRCS` 的檔案 |
|:---|:---|
| 連續串流，INT8（預設） | `app_audio_stream_int8.c` + `model_runner_int8.cc` + `model_data_int8.cc` + `logmel_features_int8.c` + `memory_monitor.c` |
| 按鈕觸發，I2S，INT8 | 把 `app_audio_stream_int8.c` 換成 `app_audio_infer_int8.c` |
| 按鈕觸發，PDM，INT8 | 把 `app_audio_stream_int8.c` 換成 `app_audio_infer_pdm_int8.c` |
| 按鈕觸發，float32 | `app_audio_infer.c` 或 `app_audio_infer_pdm.c`，並改用 `model_runner.cc`、`model_data.cc`、`logmel_features.c` |

### 4. 串流版的設定

在 `main/app_audio_stream_int8.c` 開頭可以調整：

| 巨集 | 預設 | 說明 |
|:---|:---:|:---|
| `STREAM_SAVE_MODE` | `SAVE_LOW_CONFIDENCE` | `SAVE_NONE`：不存檔；`SAVE_LOW_CONFIDENCE`：預測機率低於門檻才存；`SAVE_ALL`：每個視窗都存 |
| `LOW_CONF_THRESHOLD` | `0.5` | 低信心存檔的門檻 |
| `PCM_GAIN` | `30` | PCM 數位增益。**這是模型輸入契約的一部分，不要單獨修改** |
| `STATS_PERIOD_WINDOWS` | `30` | 每 30 個視窗（1 分鐘）印一次統計 |

沒有插 SD 卡時，推論仍然照常進行，只是不存檔。

### 5. 輸出

**序列埠**：每個視窗一行，格式如下：

```
w<編號> <預測類別> <機率> [<background> <drone> <helicopter>] logmel=<ms> cnn=<ms> wait logmel/cnn=<ms>/<ms> latency=<ms> ovf=<溢位次數>[ SAVE]
```

每分鐘另外印一次 `STATS`：處理視窗數、丟棄數、I2S 溢位數、存檔數、前處理與 CNN 的平均／最大耗時、各任務剩餘 stack、heap 最低剩餘量。

**SD 卡**：

- `S0000001.WAV`、`S0000002.WAV`…：16 kHz、16-bit、單聲道、2 秒。內容就是模型實際吃到的 PCM（已乘上增益），可以直接拿回 PC 重現結果。重新開機後會接續編號，不會覆蓋舊檔。
- `STREAM.CSV`：每個存下的視窗一行，欄位是 `file,window,pred,background,drone,helicopter`。

> FATFS 沒開長檔名，所以檔名維持 8.3 格式。

---

## 前處理規格（部署契約）

Python 訓練和 C 韌體必須用**完全相同**的前處理，下面這些值會寫進 `preprocess_config.json`，再由轉換器產生 `preprocess_config_int8.h`：

| 項目 | 值 |
|:---|:---|
| 取樣率 | 16 kHz，單聲道 |
| 視窗長度 | 2 秒 = 32000 samples |
| 麥克風轉換 | 32-bit slot → `(int16)(raw >> 16) × 30`，飽和處理 |
| 處理順序 | PCM float → 去 DC → Hamming 視窗 STFT power → HTK Mel 濾波器組 → `10·log10` → 以樣本最大值為 0 dB 參考 → 截斷在 −80 dB |
| FFT | `n_fft = 512`、`hop = 256`、不補零（124 frames） |
| Mel | 32 個頻帶，目前模型使用 **250–2500 Hz** |
| CMVN | 每個 Mel 頻帶沿時間軸減平均、除以標準差（std 下限 2 dB） |
| 模型輸入 | `[1, 32, 124, 1]` int8，NHWC，順序是 `[mel][frame][channel]` |
| 量化 | 輸入 `scale = 0.0455`、`zero_point = 15`；輸出 `scale = 1/256`、`zero_point = −128` |

前處理全程用 float32 計算，只有最後送進模型的張量才量化成 int8，這樣 Python 和 C 的數值才能對得上。

> ⚠️ `model_runner_int8.cc`、`model_data_int8.cc`、`logmel_features_int8.c`、`preprocess_config_int8.h` 是**同一份部署契約**。換模型時請用轉換器一起重新產生，不要手動修改產生出來的檔案。

---

## 模型架構

`logmel_cnn.build_logmel_cnn()`：只用 TFLite Micro 標準運算子的小型 CNN，INT8 模型約 13 KB。

```
Input (32 × 124 × 1)
Conv2D 16, 3×3            + BN + ReLU → MaxPool 2×2
SeparableConv2D 32, 3×3   + BN + ReLU → MaxPool 2×2
SeparableConv2D 48, 3×3   + BN + ReLU
AveragePooling (1 × 31)       ← 只沿時間軸平均，保留頻率位置
Conv2D 3, kernel (8 × 1)      ← 頻率感知分類層
GlobalAveragePooling → Softmax (3 類)
```

用到的運算子：`CONV_2D`、`DEPTHWISE_CONV_2D`、`MAX_POOL_2D`、`AVERAGE_POOL_2D`、`MEAN`、`SOFTMAX`。

---

## Python 環境

建議使用 Python 3.11：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-logmel.txt
```

`requirements-logmel.txt` 包含 numpy、scipy、scikit-learn、matplotlib、tensorflow 2.20、keras 3.14。

有些腳本會另外用到：

- `soundfile`：`apply_wav_gain_folder.py`、`invert_negative_dc_wavs.py`、`tools/split_wav.py`
- **gcc**（需在 PATH 中，Windows 可用 MSYS2 / MinGW-w64）：所有 Python／C 比對腳本和 `run_c_logmel_int8_inference.py` 都會即時編譯韌體的 C 原始碼

---

## 資料集準備

### 標籤規則

標籤由**檔名**判斷（`audio_dataset.infer_label`，不分大小寫），依下列順序比對：

| 標籤 | 檔名包含 | 或檔名開頭是 |
|:---|:---|:---:|
| `background` | `background`、`back`、`noise` | `b` |
| `helicopter` | `helicopter`、`heli` | `h` |
| `drone` | `drone`、`uav` | `d` |

例如 `DRONE_012.wav`、`helicopter_3.wav`、`background_a.wav`。無法判斷標籤的檔案會被略過。可以用 `tools/tiltle_edit.py` 批次改檔名。

### 建議的資料夾結構

各實驗腳本預設的結構如下（每支麥克風一個資料夾）：

```
raw_audio/
└── new_16kHz/
    ├── ics/   inmp/   pdm/           ← 訓練用（A 組錄音）
    └── test/
        ├── ICS/   INMP/   PDM/       ← 測試用（B 組錄音）
```

- 不是 16 kHz 的 WAV 會自動重新取樣。
- 長錄音會被切成不重疊的 2 秒片段；**切分 train/val 時以原始檔案為單位**，同一個錄音檔的片段不會同時出現在 train 和 val。

---

## 訓練模型

一般使用 `train_cnn_from_wav_logmel_cmvn.py`，它會自動帶入 CMVN 版本的預設值（`--input-channels cmvn`、增益 −20～+6 dB、背景混音 SNR 5～20 dB），命令列給的參數會優先採用。

```powershell
python train_cnn_from_wav_logmel_cmvn.py `
  --train-dir raw_audio/new_16kHz/ics raw_audio/new_16kHz/pdm `
  --test-dir  raw_audio/new_16kHz/test/ICS raw_audio/new_16kHz/test/INMP raw_audio/new_16kHz/test/PDM `
  --val-split 0.2 `
  --max-duration 2 `
  --f-min 250 --f-max 2500 `
  --augmentation-copies 3 `
  --epochs 50 `
  --output-dir cnn_output/logmel_cmvn/my_run
```

常用參數：

| 參數 | 預設 | 說明 |
|:---|:---:|:---|
| `--train-dir` / `--test-dir` | 必填 | 可以給多個資料夾 |
| `--val-split` | 0.2 | 從 train 資料夾撥給 validation 的原始檔比例 |
| `--max-duration` | 2.0 | 片段長度（秒）；韌體固定 2 秒 |
| `--f-min` / `--f-max` | 0 / Nyquist | Mel 頻帶範圍 |
| `--n-mels` / `--n-fft` / `--hop-length` | 32 / 512 / 256 | 改了的話韌體也要對應修改 |
| `--input-channels` | `cmvn` | 韌體支援 `logmel`、`logmel-delta`、`cmvn`；其他模式只供 PC 實驗 |
| `--mask-mel-bands` | 無 | 把指定的 Mel 列設為 0，例如 `18-31` |
| `--epochs` / `--batch-size` / `--learning-rate` | 80 / 32 / 5e-4 | |
| `--early-stopping-patience` | 50 | |
| `--grouped-cross-mic-split` | 關 | 依內容 ID 跨麥克風切分 |
| `--load-model` | 無 | 載入現有模型，跳過訓練（只匯出） |
| `--skip-tflite` | 關 | 不匯出 TFLite |

增強相關參數（`--augmentation-*`）請執行 `python train_cnn_from_wav_logmel.py -h` 查看。

### 輸出內容（`--output-dir`）

| 檔案 | 說明 |
|:---|:---|
| `logmel_cnn_classifier.keras` | 最後一個 epoch、BN 再校正後的模型 |
| `logmel_cnn_classifier_best_val_loss.keras` | val loss 最低的模型（另有 `best_val_loss/` 子資料夾放它的報告） |
| `logmel_cnn_classifier_float32.tflite` | float32 TFLite |
| `logmel_cnn_classifier_int8.tflite` | **full-INT8 TFLite（部署用）** |
| `preprocess_config.json` | 前處理參數、正規化統計值、類別名稱（轉換器需要） |
| `metrics.json` | 各 split 的指標 |
| `*_confusion_matrix.csv`、`*_classification_report.csv` | 混淆矩陣與分類報告（測試集另外按來源資料夾分開統計） |
| `training_history.png` | 訓練曲線 |
| `error/<split>/` | 被分錯的樣本 WAV |
| `original_file_split.json` | 哪些原始檔分到 train / val |

如果只有 Keras 模型，可以用 `quantize_logmel_cnn_int8.py --keras-model ... --config ...` 補做 INT8 量化。

---

## 匯出模型到韌體

```powershell
python tools\convert_logmel_tflite_to_c_int8.py `
  --model  cnn_output\logmel_cmvn\my_run\logmel_cnn_classifier_int8.tflite `
  --config cnn_output\logmel_cmvn\my_run\preprocess_config.json `
  --out-dir main
```

這會覆寫：

- `main/model_data_int8.cc` / `.h`
- `main/preprocess_config_int8.h`
- `main/logmel_conversion_summary_int8.json`（記錄來源模型和 SHA-256，方便確認韌體裡是哪一個模型）

接著重新 `idf.py build flash` 即可。

---

## 評估與 Python／C 一致性驗證

### 在 WAV 資料夾上評估

```powershell
python evaluate_trained_audio_folders.py `
  --input-dirs raw_audio/new_16kHz/test/ICS raw_audio/new_16kHz/test/INMP raw_audio/new_16kHz/test/PDM `
  --model-dir cnn_output/logmel_cmvn/my_run `
  -o evaluation_output/my_run
```

加上 `--compare-model <int8.tflite>` 可以在同一批片段上同時跑第二個模型，報告準確率差異、預測一致率和機率誤差。

### 檢查 C 前處理和 Python 是否一致

```powershell
# 單一 WAV：逐階段比對 CMVN、int8 輸入和 TFLite 輸出
python compare_logmel_python_c_int8.py --wav some.wav --config cnn_output/logmel_cmvn/my_run/preprocess_config.json
```

預設容許誤差：正規化值 0.02、量化值 ±1、輸出機率 0.02。

### 在 PC 上重現裝置推論

用韌體自己的 `logmel_features_int8.c`（以 gcc 編譯）加上 `model_data_int8.cc` 裡嵌入的 INT8 模型跑 WAV。從 SD 卡拿回來的 16 kHz WAV 會原封不動送進去，結果應該和裝置逐位元相同：

```powershell
python run_c_logmel_int8_inference.py --input E:\ -o c_int8_inference_output
```

### 裝置 debug dump（DID_NODE / infer_node）

部署在 `DID_NODE/infer_node` 的節點可以開 `INFER_AUDIO_DEBUG_DUMP=1`，把前幾個視窗的 PCM、各階段特徵和每一層 CNN 張量 dump 出來：

```powershell
python tools/receive_infer_node_dump.py --port COM14 --reset   # 先關掉 idf.py monitor
python compare_infer_node_debug_dump.py --dump-dir infer_node_dump
```

---

## 比較實驗腳本

| 腳本 | 實驗內容 |
|:---|:---|
| `run_mic_loo_comparison.py` | 每支麥克風（ICS / INMP / PDM）的 A 組資料訓練，測試所有麥克風的 B 組資料（3×3 格），比較 Xu MFCC-DNN 和 Log-Mel CMVN CNN。 |
| `run_ltcmn_mic_loo_comparison.py` | 同樣的協定，比較 per-clip CMVN 和長期 CMN（`ltcmn_bg`、`ltcmn_all`、`cmvn_ltcmn_bg`），附 95% t 信賴區間和 Wilcoxon 檢定。 |
| `run_xu_sample_rate_check.py` | 同一批 44.1 kHz 錄音，分別用 44.1 kHz 和 16 kHz 跑 Xu 的流程，看取樣率是否影響準確率。 |

這些腳本都支援 `--seeds` 跑多個隨機種子，以及 `--summary-only` 只重新整理已有的結果。

---

## 注意事項與常見問題

- **一定要有 PSRAM**：開機時如果出現 `Pipeline buffer allocation failed` 或 `INT8 model initialization failed`，先確認板子有 PSRAM，且 `sdkconfig` 中 `CONFIG_SPIRAM=y`。
- **不要單獨改 `PCM_GAIN` 或 PCM 轉換方式**：訓練資料就是用 `(raw >> 16) × 30` 錄的，改了會讓模型輸入分布跑掉。注意 DID_NODE 的 PDM 路徑讀的是 16-bit slot、沒有 `>> 16`，和本專案不同。
- **I2S 溢位或丟視窗**：序列埠出現 `All 4 windows busy` 或 `ovf` 不是 0，代表前處理／推論／SD 寫入跟不上。可以把 `STREAM_SAVE_MODE` 改成 `SAVE_NONE` 確認是不是 SD 卡太慢。
- **SD 卡掛載失敗**：確認卡片是 FAT 格式、接線正確（SDMMC 1-bit：CLK 14、CMD 15、D0 2）。韌體不會自動格式化。
- **換到 ESP32-S3（例如 DID_NODE）時**：ESP-NN 的 S3 組合語言 avg-pool 在輸出不是正方形時會算錯（本模型的 `time_average_pool` 輸出是 8×1×48）。infer_node 已在 `pooling.cc` 強制改用 ANSI 版本；如果更新了 esp-nn 或移除這個修正，請用 debug dump 重新驗證。
- **燒錄失敗**：先用 `idf.py -p PORT monitor` 確認板子有輸出；必要時在 `menuconfig` 降低下載鮑率。
