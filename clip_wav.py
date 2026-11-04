from pathlib import Path
import wave

def trim_wav_edges(
    folder_path: str,
    trim_start_sec: float = 1.0,
    trim_end_sec: float = 1.0,
    recursive: bool = False,
    dry_run: bool = True
):
    """
    將資料夾中所有 wav/WAV 檔案的前 N 秒與後 M 秒剪掉，並覆蓋原檔。
    """

    folder = Path(folder_path)

    if not folder.exists():
        print(f"[ERROR] 資料夾不存在：{folder}")
        return

    if not folder.is_dir():
        print(f"[ERROR] 這不是資料夾：{folder}")
        return

    all_files = folder.rglob("*") if recursive else folder.glob("*")

    for wav_path in all_files:
        if not wav_path.is_file():
            continue

        # 接受大小寫不同的 wav 副檔名
        if wav_path.suffix.lower() != ".wav":
            continue

        try:
            with wave.open(str(wav_path), "rb") as wf:
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                comptype = wf.getcomptype()
                compname = wf.getcompname()

                # 計算前後要裁切的 frame 數量
                start_frames = int(trim_start_sec * framerate)
                end_frames = int(trim_end_sec * framerate)
                keep_frames = nframes - start_frames - end_frames
                
                duration = nframes / framerate

                # 如果音檔總長度比要裁切掉的秒數還短，則跳過
                if keep_frames <= 0:
                    print(f"[SKIP] {wav_path.name} 總長度 {duration:.2f}s，不足以裁切前 {trim_start_sec}s 與後 {trim_end_sec}s")
                    continue

                # 將讀取指標移動到起始點，並讀取要保留的 frame
                wf.setpos(start_frames)
                kept_frames = wf.readframes(keep_frames)

            new_duration = keep_frames / framerate

            if dry_run:
                print(
                    f"[PREVIEW] {wav_path.name}: "
                    f"原本 {duration:.2f}s -> 變成 {new_duration:.2f}s "
                    f"(擷取 {trim_start_sec}s 到 {duration - trim_end_sec:.2f}s 的片段)"
                )
                continue

            # 寫入覆蓋原檔 (注意：這裡是覆蓋，建議做好備份)
            with wave.open(str(wav_path), "wb") as wf_out:
                wf_out.setnchannels(nchannels)
                wf_out.setsampwidth(sampwidth)
                wf_out.setframerate(framerate)
                wf_out.setcomptype(comptype, compname)
                wf_out.writeframes(kept_frames)

            print(f"[DONE] {wav_path.name}: {duration:.2f}s -> {new_duration:.2f}s")

        except wave.Error as e:
            print(f"[ERROR] WAV 格式錯誤：{wav_path.name} ({e})")

        except Exception as e:
            print(f"[ERROR] 處理失敗：{wav_path.name} ({e})")


if __name__ == "__main__":
    folder_path = r"raw_audio\pdm\pdm_real"

    # 先使用 dry_run=True 進行預覽，確認秒數計算無誤
    trim_wav_edges(
        folder_path=folder_path,
        trim_start_sec=2.0,  # 前面切掉 1 秒
        trim_end_sec=0.0,    # 後面切掉 1 秒
        recursive=False,
        dry_run=False         # 確認無誤後，把這裡改成 False 才會真正寫入檔案
    )