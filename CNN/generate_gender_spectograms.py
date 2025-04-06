import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

CLIPS_DIR = "../data/common_voice/cv-corpus-21.0-delta-2025-03-14/en/clips"
TSV_VALIDATED = "../data/common_voice/cv-corpus-21.0-delta-2025-03-14/en/validated.tsv"
TSV_INVALIDATED = "../data/common_voice/cv-corpus-21.0-delta-2025-03-14/en/invalidated.tsv"
OUTPUT_DIR = "processed_spectrograms"
SAMPLE_RATE = 16000
N_MELS = 128
CSV_OUTPUT_PATH = "gender_labels.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df_valid = pd.read_csv(TSV_VALIDATED, sep='\t')
df_invalid = pd.read_csv(TSV_INVALIDATED, sep='\t')

print(f"Validated rows: {len(df_valid)}")
print(f"Invalidated rows: {len(df_invalid)}")

df = pd.concat([df_valid, df_invalid], ignore_index=True)
df = df.drop_duplicates(subset='path')
print(f"Combined and deduplicated rows: {len(df)}")

df = df.dropna(subset=['gender'])
df['gender'] = df['gender'].str.strip().str.lower()

def map_gender(value):
    if value.startswith("male"):
        return 0
    elif value.startswith("female"):
        return 1
    return None

df['gender_label'] = df['gender'].apply(map_gender)
df = df.dropna(subset=['gender_label'])
df['gender_label'] = df['gender_label'].astype(int)

print(f"After gender filtering: {len(df)}")
print("Final gender label counts:", df['gender_label'].value_counts().to_dict())

# === 4. Convert each file to spectrogram ===
npy_paths, valid_paths, labels = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting to spectrograms"):
    filename = row['path']
    clip_path = os.path.join(CLIPS_DIR, filename)
    npy_path = os.path.join(OUTPUT_DIR, filename.replace('.mp3', '.npy'))

    try:
        y, sr = librosa.load(clip_path, sr=SAMPLE_RATE)
        y, _ = librosa.effects.trim(y)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        log_S = librosa.power_to_db(S, ref=np.max)
        np.save(npy_path, log_S)

        npy_paths.append(npy_path)
        valid_paths.append(filename)
        labels.append(row['gender_label'])

    except Exception as e:
        print(f"Failed on {filename}: {e}")

label_df = pd.DataFrame({
    "path": valid_paths,
    "gender_label": labels,
    "npy_path": npy_paths
})
label_df.to_csv(CSV_OUTPUT_PATH, index=False)

print(f"Saved {CSV_OUTPUT_PATH} with {len(label_df)} usable rows.")
