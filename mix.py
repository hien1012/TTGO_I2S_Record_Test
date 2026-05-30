import shutil
import random
from pathlib import Path

def mix_and_split_dataset_clean_name(
    source_dirs: list, 
    dest_root: str, 
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    從多個麥克風資料夾中讀取檔案，依據檔名前綴分類，混合並切割為 Train / Val。
    檔名精簡化：原檔名_麥克風簡寫.wav (不含隨機碼)
    """
    random.seed(seed)
    dest_root_path = Path(dest_root)

    train_dir = dest_root_path / "train"
    val_dir = dest_root_path / "val"

    # 定義麥克風資料夾名稱到簡寫的對應表
    mic_name_mapping = {
        "ics43434": "ics",
        "inmp441": "inmp",
        "pdm": "pdm"
    }

    class_files = {}

    # 1. 遍歷所有來源資料夾，收集並分類檔案
    for src_dir in source_dirs:
        src_path = Path(src_dir)
        if not src_path.exists() or not src_path.is_dir():
            print(f"[警告] 找不到資料夾，已跳過: {src_path}")
            continue

        all_wavs = list(src_path.glob("*.[wW][aA][vV]"))
        
        for file_path in all_wavs:
            file_stem = file_path.stem
            
            if '_' not in file_stem:
                print(f"[警告] 檔名格式不符，已略過: {file_path.name}")
                continue
                
            class_name = file_stem.split('_')[0].lower()

            if class_name not in class_files:
                class_files[class_name] = []
            class_files[class_name].append(file_path)

    # 2. 建立目標資料夾並進行混合、打亂與切割複製
    for class_name, files in class_files.items():
        if not files:
            continue

        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)

        # 按編號分組，確保同一編號的所有麥克風變體在同一集合中
        file_groups = {}
        for file_path in files:
            # 提取編號 (例如 DRONE_1 from DRONE_1_ics.wav)
            parts = file_path.stem.split('_')
            if len(parts) >= 2:
                # 取前兩個部分作為編號 (例如 DRONE_1)
                file_id = '_'.join(parts[:2])
            else:
                file_id = file_path.stem
            
            if file_id not in file_groups:
                file_groups[file_id] = []
            file_groups[file_id].append(file_path)

        # 對編號進行隨機分配
        file_ids = list(file_groups.keys())
        train_ids = []
        val_ids = []
        
        # 依據比例隨機分配每個編號
        for file_id in file_ids:
            if random.random() < train_ratio:
                train_ids.append(file_id)
            else:
                val_ids.append(file_id)

        # 收集該集合中的所有檔案
        train_files = []
        val_files = []
        for file_id in train_ids:
            train_files.extend(file_groups[file_id])
        for file_id in val_ids:
            val_files.extend(file_groups[file_id])

        print(f"處理類別 [{class_name.upper()}] - 總計: {len(files)} 筆 "
              f"({len(file_ids)} 個編號) (Train: {len(train_files)}, Val: {len(val_files)})")

        def copy_and_rename(file_list, target_folder):
            for file_path in file_list:
                # 取得來源麥克風資料夾名稱 (例如 'inmp441')
                raw_mic_name = file_path.parent.name
                
                # 從字典中查找簡寫，如果找不到就用原名稱
                short_mic = mic_name_mapping.get(raw_mic_name, raw_mic_name)
                
                # 組合精簡版新檔名: 原檔名_簡寫.wav (例如 DRONE_1_inmp.wav)
                new_filename = f"{file_path.stem}_{short_mic}{file_path.suffix.lower()}"
                
                dest_path = target_folder / new_filename
                
                shutil.copy2(file_path, dest_path)

        # 執行複製
        copy_and_rename(train_files, train_dir)
        copy_and_rename(val_files, val_dir)

    print(f"\n[完成] 資料已成功混合並切割至: {dest_root_path.absolute()}")

# ==========================================
# 執行區塊
# ==========================================
if __name__ == "__main__":
    base_path = r"C:\Users\Potato\OneDrive\Pictures\Desktop\碩論資料\測試階段\microphone\PC_AUDIO_DATA"
    
    SOURCES = [
        f"{base_path}\\ics",
        f"{base_path}\\inmp",
        f"{base_path}\\pdm"
    ]
    
    DESTINATION = f"{base_path}\\mix_2"
    
    mix_and_split_dataset_clean_name(
        source_dirs=SOURCES,
        dest_root=DESTINATION,
        train_ratio=0.6
    )