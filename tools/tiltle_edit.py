from pathlib import Path
import re


def rename_files_keep_number(
    folder_path: str,
    new_name: str,
    file_ext: str = ".wav",
    dry_run: bool = True
):
    """
    將資料夾內指定副檔名的檔案重新命名：
    忽略原本檔名中的文字，只保留數字，並替換成新的文字前綴。

    例如：
    1.wav              -> HELICOPTER_1.wav
    HELICOPTER_1.wav   -> DRONE_1.wav
    test_023.wav       -> DRONE_023.wav

    參數：
    folder_path : 資料夾路徑
    new_name    : 新的文字前綴，例如 "DRONE"
    file_ext    : 要處理的副檔名，例如 ".wav"
    dry_run     : True 只預覽，不真的改名；False 才會真的改名
    """

    folder = Path(folder_path)

    if not folder.exists():
        print(f"[ERROR] 資料夾不存在：{folder}")
        return

    if not folder.is_dir():
        print(f"[ERROR] 這不是資料夾：{folder}")
        return

    file_ext = file_ext.lower()

    for file in folder.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() != file_ext:
            continue

        stem = file.stem  # 不含副檔名的檔名，例如 HELICOPTER_1

        # 找出檔名中的數字
        match = re.search(r"\d+", stem)

        if not match:
            print(f"[SKIP] 找不到數字：{file.name}")
            continue

        number = match.group()

        new_filename = f"{new_name}_{number}{file.suffix}"
        new_path = file.with_name(new_filename)

        if new_path.exists():
            print(f"[SKIP] 目標檔案已存在：{new_filename}")
            continue

        if dry_run:
            print(f"[PREVIEW] {file.name}  ->  {new_filename}")
        else:
            file.rename(new_path)
            print(f"[RENAMED] {file.name}  ->  {new_filename}")


if __name__ == "__main__":
    folder_path = r"C:\Users\Potato\Downloads\Groundtruth by iPhone"
    new_name = "DRONE_"

    # 第一次建議先用 dry_run=True 預覽
    rename_files_keep_number(
        folder_path=folder_path,
        new_name=new_name,
        file_ext=".wav",
        dry_run=False
    )

