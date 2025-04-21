import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

# === CONFIG ===
CLIPS_DIR = "../data/cv-corpus-12.0-delta-2022-12-07/en/clips"
TSV_VALIDATED = "../data/cv-corpus-12.0-delta-2022-12-07/en/validated.tsv"
TSV_INVALIDATED = "../data/cv-corpus-12.0-delta-2022-12-07/en/invalidated.tsv"
TSV_OTHER = "../data/cv-corpus-12.0-delta-2022-12-07/en/other.tsv"
OUTPUT_DIR = "processed_spectrograms"
CSV_OUTPUT_PATH = "labels.csv"
SAMPLE_RATE = 16000
N_MELS = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the TSV files
df_valid = pd.read_csv(TSV_VALIDATED, sep='\t')
df_invalid = pd.read_csv(TSV_INVALIDATED, sep='\t')
df_other = pd.read_csv(TSV_OTHER, sep='\t')

print(f"Validated rows: {len(df_valid)}")
print(f"Invalidated rows: {len(df_invalid)}")
print(f"Other rows: {len(df_other)}")

df = pd.concat([df_valid, df_invalid, df_other], ignore_index=True)
df = df.drop_duplicates(subset='path')
print(f"Combined and deduplicated rows: {len(df)}")

# Remove any examples without a gender or age label
df = df.dropna(subset=['gender', 'age'])

df['gender'] = df['gender'].str.strip().str.lower()
df['age'] = df['age'].str.strip().str.lower()

def map_gender(value):
    if value.startswith("male"):
        return 0
    elif value.startswith("female"):
        return 1
    return None

def map_age(value):
    buckets = {
        'teens': 0,
        'twenties': 1,
        'thirties': 2,
        'fourties': 3,
        'fifties': 4,
        'sixties': 5,
        'seventies': 6,
        'eighties': 7,
        'nineties': 8,
    }
    return buckets.get(value, None)

df['gender_label'] = df['gender'].apply(map_gender)
df['age_label'] = df['age'].apply(map_age)

df = df.dropna(subset=['gender_label', 'age_label'])
df['gender_label'] = df['gender_label'].astype(int)
df['age_label'] = df['age_label'].astype(int)

#Remove class 7 since it only has 1 member
index_to_drop = df[df['age_label'] == 7].index
df = df.drop(index_to_drop)

print(f"After filtering: {len(df)}")
print("Gender label distribution:", df['gender_label'].value_counts().to_dict())
print("Age label distribution:", df['age_label'].value_counts().to_dict())

# Generate a spectogram for each .mp3 file
npy_paths, valid_paths, gender_labels, age_labels = [], [], [], []

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
        gender_labels.append(row['gender_label'])
        age_labels.append(row['age_label'])

    except Exception as e:
        print(f"Failed on {filename}: {e}")

# Save to labels.csv
label_df = pd.DataFrame({
    "path": valid_paths,
    "npy_path": npy_paths,
    "gender_label": gender_labels,
    "age_label": age_labels
})
label_df.to_csv(CSV_OUTPUT_PATH, index=False)

print(f"Saved {CSV_OUTPUT_PATH} with {len(label_df)} usable rows.")
