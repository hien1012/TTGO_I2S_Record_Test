import os
import random
import shutil

def split_dataset():
    # 1. 設定基礎路徑 (請確保腳本與 total_audio 資料夾在同一層)
    base_dir = 'data_train'
    train_dir = os.path.join(base_dir, 'inmp_pdm_ics/train')
    val_dir = os.path.join(base_dir, 'inmp_pdm_ics/val')

    # 2. 設定類別與抽取比例
    categories = ['DRONE', 'BACKGROUND', 'HELICOPTER']
    split_ratio = 0.3

    # 3. 確保 val 資料夾內有這三個類別的子資料夾
    for cat in categories:
        os.makedirs(os.path.join(val_dir, cat), exist_ok=True)

    if not os.path.exists(train_dir):
        print(f"找不到 {train_dir} 目錄，請確認路徑是否正確。")
        return

    # 取得 train 底下的子資料夾 (ics, ics_1, pdm, pdm_1)
    subfolders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(train_dir, subfolder)
        
        # 建立字典分類檔案
        files_by_category = {cat: [] for cat in categories}

        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith('.wav'):
                upper_name = filename.upper()
                for cat in categories:
                    if cat in upper_name:
                        files_by_category[cat].append(filename)
                        break

        print(f"--- 正在處理資料夾: {subfolder} ---")

        for cat, files in files_by_category.items():
            total_files = len(files)
            val_count = int(total_files * split_ratio)

            print(f"  類別 [{cat}]: 總數 {total_files} -> 抽取 {val_count} 個至 val")

            if val_count > 0:
                sampled_files = random.sample(files, val_count)

                for f in sampled_files:
                    src_path = os.path.join(subfolder_path, f)
                    
                    # --- 修改的區塊 ---
                    # 將檔名與副檔名分離 (例如: 'DRONE_1', '.wav')
                    name, ext = os.path.splitext(f)
                    
                    # 重新組合: 原檔名_資料夾名稱.wav (例如: DRONE_1_ics.wav)
                    new_filename = f"{name}_{subfolder}{ext}"
                    # -----------------
                    
                    dst_path = os.path.join(val_dir, cat, new_filename)

                    # 執行移動 (若想保留原檔，請將 shutil.move 改為 shutil.copy)
                    shutil.move(src_path, dst_path)

    print("\n✅ 資料抽取與搬移完成！")

if __name__ == "__main__":
    random.seed(42) 
    split_dataset()