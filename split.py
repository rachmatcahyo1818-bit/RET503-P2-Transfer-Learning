import shutil
import random
from pathlib import Path
SEED=42
VAL_RATIO=0.2
ROOT=Path('dataset_corrected')
OUT=Path('dataset')
random.seed(SEED)
for cls in [d for d in ROOT.iterdir() if d.is_dir()]:
    images=[p for p in cls.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
    random.shuffle(images)
    val_count=max(1,round(len(images)*VAL_RATIO))
    val_images=images[:val_count]
    train_images=images[val_count:]
    for split,items in [('train',train_images),('val',val_images)]:
        out_dir=OUT/split/cls.name
        out_dir.mkdir(parents=True,exist_ok=True)
        for img in items:
            shutil.copy2(img,out_dir/img.name)
    print(cls.name+': train='+str(len(train_images))+', val='+str(len(val_images)))
print('Split dataset selesai.')
