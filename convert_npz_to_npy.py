# convert_npz_to_npy.py
import numpy as np

data = np.load("embeddings.npz")

X = data["embeddings"]
y = data["labels"]

np.save("X_train.npy", X)
np.save("y_train.npy", y)

print("Berhasil membuat X_train.npy dan y_train.npy")
print("Shape X:", X.shape)
print("Jumlah label:", len(y))
