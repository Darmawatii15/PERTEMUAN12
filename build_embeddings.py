# build_embeddings.py
import os
import numpy as np
from utils_facenet import embed_from_path

DATA_DIR = "data/train"

embeddings = []
labels = []

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"Proses folder: {person}")

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        print("  →", img_path)

        emb = embed_from_path(img_path)
        if emb is None:
            print("  [X] Wajah tidak terdeteksi. Dilewati.")
            continue

        embeddings.append(emb)
        labels.append(person)

embeddings = np.array(embeddings)
labels = np.array(labels)

print("\nJumlah embedding:", len(embeddings))

np.savez("embeddings.npz", embeddings=embeddings, labels=labels)
print("embeddings.npz berhasil disimpan!")
