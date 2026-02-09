import os
import shutil
from pathlib import Path

visdrone_base = Path("C:/Users/shabd/Documents/AURORA/VisDrone2019-DET-train")
images_dir = visdrone_base / "images"
annotations_dir = visdrone_base / "annotations"
llvip_person_dir = Path("C:/Users/shabd/Documents/AURORA/dataset/train/person")

output_base = Path("C:/Users/shabd/Documents/AURORA/dataset")
person_train = output_base / "train" / "person"
no_person_train = output_base / "train" / "no_person"

# Clear existing directories
if person_train.exists():
    shutil.rmtree(person_train)
if no_person_train.exists():
    shutil.rmtree(no_person_train)

person_train.mkdir(parents=True, exist_ok=True)
no_person_train.mkdir(parents=True, exist_ok=True)

person_count = 0
no_person_count = 0
max_no_person = 5500  # Collect all available no_person first

# Add VisDrone images - collect all no_person first
for ann_file in annotations_dir.glob("*.txt"):
    if no_person_count >= max_no_person:
        break
    
    img_name = ann_file.stem + ".jpg"
    img_path = images_dir / img_name
    
    if not img_path.exists():
        continue
    
    with open(ann_file, 'r') as f:
        lines = f.readlines()
    
    has_person = any(line.split(',')[5] == '1' for line in lines if line.strip())
    
    if not has_person:
        dest = no_person_train / img_name
        if not dest.exists():
            shutil.copy(img_path, dest)
        no_person_count += 1
        if no_person_count % 500 == 0:
            print(f"VisDrone no_person: {no_person_count}")

print(f"Collected {no_person_count} no_person images")
max_person = no_person_count  # Match person count to no_person

# Now add person images to match
for ann_file in annotations_dir.glob("*.txt"):
    if person_count >= max_person // 2:
        break
    
    img_name = ann_file.stem + ".jpg"
    img_path = images_dir / img_name
    
    if not img_path.exists():
        continue
    
    with open(ann_file, 'r') as f:
        lines = f.readlines()
    
    has_person = any(line.split(',')[5] == '1' for line in lines if line.strip())
    
    if has_person:
        dest = person_train / img_name
        if not dest.exists():
            shutil.copy(img_path, dest)
        person_count += 1
        if person_count % 500 == 0:
            print(f"VisDrone person: {person_count}")

# Add LLVIP to match remaining
llvip_count = 0
max_llvip = max_person - person_count
for img_path in llvip_person_dir.glob("*.jpg"):
    if llvip_count >= max_llvip:
        break
    dest = person_train / img_path.name
    if not dest.exists():
        shutil.copy(img_path, dest)
    llvip_count += 1
    person_count += 1
    if llvip_count % 500 == 0:
        print(f"LLVIP person: {llvip_count}")

print(f"\nFinal balanced dataset:")
print(f"Person: {person_count}")
print(f"No person: {no_person_count}")
print(f"Ratio: {person_count/no_person_count:.2f}:1")
print(f"\nModel will learn: people in bright AND low-light conditions")
