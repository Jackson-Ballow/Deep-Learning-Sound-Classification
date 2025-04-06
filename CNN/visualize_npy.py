import numpy as np
import matplotlib.pyplot as plt

spec = np.load("processed_spectrograms/common_voice_en_41917197.npy")

plt.figure(figsize=(10, 4))
plt.imshow(spec, origin='lower', aspect='auto', cmap='magma')
plt.colorbar(label='dB')
plt.title("Mel Spectrogram")
plt.tight_layout()

# Save to file instead of showing
plt.savefig("spectogram_preview/spectrogram_preview.png")
print("Saved to spectrogram_preview.png")
