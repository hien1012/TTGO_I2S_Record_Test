import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 讀取 WAV 檔案
filename = 'REC.wav'
sample_rate, data = wavfile.read(filename)

print(f"Sample Rate: {sample_rate} Hz")
print(f"Data Length: {len(data)} samples ({len(data)/sample_rate:.2f} seconds)")

# 繪製波形圖
plt.figure(figsize=(10, 4))
time_axis = np.linspace(0, len(data) / sample_rate, num=len(data))
plt.plot(time_axis, data, color='blue', linewidth=0.5)
plt.title(f"Waveform of {filename}")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()