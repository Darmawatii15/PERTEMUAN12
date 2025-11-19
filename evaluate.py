# evaluate.py
import os
import numpy as np
import joblib
from utils_facenet import embed_from_path

VAL_DIR = "data/val"
clf = joblib.load("svm_model.pkl")

correct = 0
total = 0

for person in os.listdir(VAL_DIR):
    person_dir = os.path.join(VAL_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"Testing folder: {person}")

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        emb = embed_from_path(img_path)
        if emb is None:
            print(f"  [X] Gagal deteksi wajah: {img_name}")
            continue

        pred = clf.predict([emb])[0]
        print(f"  → {img_name} | Prediksi: {pred} | Label benar: {person}")

        if pred == person:
            correct += 1
        total += 1

print("\nTotal data:", total)
print("Benar:", correct)
print("Akurasi:", correct / total if total > 0 else 0)
