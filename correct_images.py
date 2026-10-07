from pathlib import Path
import cv2
import numpy as np

d=np.load("calib.npz")
K=d["K"]
dist=d["dist"]
root=Path("dataset_raw")
out=Path("dataset_corrected")
total=0
failed=0

for cls in [p for p in root.iterdir() if p.is_dir()]:
    out_cls=out / cls.name
    out_cls.mkdir(parents=True,exist_ok=True)
    for p in cls.iterdir():
        if p.suffix.lower() not in {".jpg",".jpeg",".png"}:
            continue
        img=cv2.imread(str(p))
        if img is None:
            failed+=1
            print("GAGAL:",p)
            continue
        corrected=cv2.undistort(img,K,dist,None,K)
        cv2.imwrite(str(out_cls / p.name),corrected)
        total+=1

print("SELESAI:",total,"gambar terkoreksi")
print("GAGAL:",failed)
