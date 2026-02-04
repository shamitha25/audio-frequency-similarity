import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Load audio files
ref_audio, sr1 = librosa.load("audio1.wav", sr=None)
pat_audio, sr2 = librosa.load("audio2.wav", sr=None)

# Make both signals same length
min_len = min(len(ref_audio), len(pat_audio))
ref_audio = ref_audio[:min_len]
pat_audio = pat_audio[:min_len]

# Time axis
time = np.linspace(0, min_len / sr1, min_len)

# Plot time-domain comparison (LIKE YOUR IMAGE)
plt.figure(figsize=(10, 5))
plt.plot(time, ref_audio, label="Reference (ref)", color="blue")
plt.plot(time, -pat_audio, label="Pattern (pat)", color="orange")  # inverted for clarity

plt.title("Audio Signal Comparison")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- Similarity Score (Frequency Domain) --------
fft_ref = np.abs(np.fft.fft(ref_audio))
fft_pat = np.abs(np.fft.fft(pat_audio))

fft_ref = fft_ref[:min_len // 2]
fft_pat = fft_pat[:min_len // 2]
# -------- Frequency Domain Plot --------
freq = np.linspace(0, sr1/2, len(fft_ref))

plt.figure(figsize=(10, 5))
plt.plot(freq, fft_ref, label="Reference FFT", color="blue")
plt.plot(freq, fft_pat, label="Pattern FFT", color="orange")

plt.title("Frequency Domain Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

similarity_score = 1 - cosine(fft_ref, fft_pat)
print(f"Similarity Score: {similarity_score:.4f}")