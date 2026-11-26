import soundfile as sf
import os
import numpy as np

# 取得目前程式碼所在的資料夾絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 利用 os.path.join 組合出正確的絕對路徑
train_input_folder  = os.path.join(BASE_DIR, "data_train/inmp_pdm_ics/train")
val_input_folder    = os.path.join(BASE_DIR, "data_train/inmp_pdm_ics/val")
input1              = os.path.join(BASE_DIR, "data_train/inmp_pdm_ics/test")

train_output_folder = os.path.join(BASE_DIR, "data_train/inmp_pdm_ics/train/split")
val_output_folder   = os.path.join(BASE_DIR, "data_train/inmp_pdm_ics/val/split")
output1             = os.path.join(BASE_DIR, "data_train/inmp_pdm_ics/test/split")

# 確保輸出資料夾存在
os.makedirs(train_output_folder,    exist_ok=True)
os.makedirs(val_output_folder,      exist_ok=True)
os.makedirs(output1,                exist_ok=True)

def split_audio(file_path, output_folder, segment_duration=2):
    labels = ['BACKGROUND', 'DRONE', 'HELICOPTER']
    new_labels = ['B', 'D', 'H']
    short_to_label = {'B': 'BACKGROUND', 'D': 'DRONE', 'H': 'HELICOPTER'}
    ignored_source_parts = set(labels + list(short_to_label.keys()))

    # 讀取原始音檔，保留 int16 格式
    y, sr = sf.read(file_path, dtype='int16')
    segment_samples = int(segment_duration * sr)

    # 取得原始檔案的上一層資料夾名稱 (例如從 'total_audio/train/ics/DRONE_1.wav' 取得 'ics')
    parent_folder_name = os.path.basename(os.path.dirname(file_path))

    # --- 修正檔名解析邏輯 ---
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    parts = base_name.split("_")
    
    # 驗證檔名格式，至少需要兩個部分 (標籤_編號)
    if len(parts) < 2:
        print(f"⚠️  檔名格式不符，已跳過: {file_path}")
        print(f"   期望格式: LABEL_NUMBER.wav (例如: BACKGROUND_1.wav)")
        return
    
    label = parts[0]
    num = parts[1]
    source_name = parent_folder_name
    
    # 驗證標籤是否有效 (不區分大小寫)
    label_upper = label.upper()
    if label_upper in short_to_label:
        label_upper = short_to_label[label_upper]
    elif label_upper not in labels:
        for part in parts:
            part_upper = part.upper()
            if part_upper in labels:
                label_upper = part_upper
                break
            if part_upper in short_to_label:
                label_upper = short_to_label[part_upper]
                break
    if label_upper not in labels:
        print(f"❌ 無效的標籤 '{label}': {file_path}")
        print(f"   允許的標籤: {labels}")
        return
    
    name_index = labels.index(label_upper)
    
    # Prefer source/mic suffix in names such as BACKGROUND_1_ics.wav or
    # BACKGROUND_1_inmp.wav. Fall back to the parent folder when absent.
    for part in parts[2:]:
        part_upper = part.upper()
        if part_upper in ignored_source_parts:
            continue
        if part_upper.startswith("SEG"):
            continue
        source_name = part
        break

    for i in range(0, len(y), segment_samples):
        segment = y[i:i + segment_samples]

        if len(segment) < segment_samples:
            print(f"丟棄不足 {segment_duration} 秒的片段: {file_path}")
            continue

        # 【修改這裡】將子資料夾名稱 (parent_folder_name) 加進新檔名中
        # 這樣就會產生如: D_1_ics_seg1.wav, D_1_ics_1_seg1.wav
        output_file_name = f"{new_labels[name_index]}_{num}_{source_name}_seg{i // segment_samples + 1}.wav"
        output_file_path = os.path.join(output_folder, output_file_name)

        # 強制寫成 PCM_16 格式
        sf.write(output_file_path, segment, sr, subtype='PCM_16')

        print(f"輸出檔案: {output_file_path}")

#  加入這段來檢查路徑
print(f" Train 資料夾絕對路徑: {os.path.abspath(train_input_folder)}")
if not os.path.exists(train_input_folder):
    print(" 錯誤：找不到 Train 輸入資料夾！請檢查相對路徑。")

# --- 處理 train 資料 ---
print(" 處理 train 資料...")
for root, dirs, files in os.walk(train_input_folder):
    dirs[:] = [
        d for d in dirs
        if os.path.abspath(os.path.join(root, d)) != os.path.abspath(train_output_folder)
    ]
    for file_name in files:
        print(f"找到檔案: {file_name}")
        if file_name.endswith((".mp3", ".wav", ".WAV")):
            file_path = os.path.join(root, file_name)
            split_audio(file_path, train_output_folder)

for root, dirs, files in os.walk(val_input_folder):
    dirs[:] = [
        d for d in dirs
        if os.path.abspath(os.path.join(root, d)) != os.path.abspath(val_output_folder)
    ]
    for file_name in files:
        print(f"找到檔案: {file_name}")
        if file_name.endswith((".mp3", ".wav", ".WAV")):
            file_path = os.path.join(root, file_name)
            split_audio(file_path, val_output_folder)
            
for root, dirs, files in os.walk(input1):
    for file_name in files:
        print(f"找到檔案: {file_name}")
        if file_name.endswith((".mp3", ".wav", ".WAV")):
            file_path = os.path.join(root, file_name)
            split_audio(file_path, output1)

print("全部切割完成！")
