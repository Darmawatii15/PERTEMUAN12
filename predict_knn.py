# predict_knn.py
import sys, joblib
from utils_facenet import embed_from_path

if len(sys.argv) < 2:
    print("Cara pakai: python predict_knn.py <gambar>")
    exit()

model = joblib.load("facenet_knn.joblib")
emb = embed_from_path(sys.argv[1])

if emb is None:
    print("Wajah tidak ditemukan")
    exit()

pred = model.predict([emb])[0]
prob = model.predict_proba([emb])[0]

print("Prediksi:", pred)
print("Probabilitas:", prob)
