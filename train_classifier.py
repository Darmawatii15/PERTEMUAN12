# train_classifier.py
import numpy as np
from sklearn.svm import SVC
import joblib

# Load embeddings
data = np.load("embeddings.npz")
X = data["embeddings"]   # embedding 512-dim
y = data["labels"]        # nama orangnya

print("Shape X:", X.shape)
print("Labels:", y)

# Create classifier
clf = SVC(kernel="linear", probability=True)

# Train
print("Training classifier...")
clf.fit(X, y)

# Save model
joblib.dump(clf, "svm_model.pkl")
print("Model SVM berhasil disimpan sebagai svm_model.pkl!")
