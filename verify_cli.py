# verify_cli.py
import argparse
from utils_facenet import embed_from_path, cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("img1")
parser.add_argument("img2")
parser.add_argument("--th", type=float, default=0.85)
args = parser.parse_args()

e1 = embed_from_path(args.img1)
e2 = embed_from_path(args.img2)

if e1 is None or e2 is None:
    print("❌ Wajah tidak terdeteksi.")
else:
    sim = cosine_similarity(e1, e2)
    print(f"Similarity = {sim:.4f}")
    print("MATCH" if sim >= args.th else "NO MATCH")
