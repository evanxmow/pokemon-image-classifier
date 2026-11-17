"""
prepare_data.py -- turn a messy downloaded dataset into a clean train/val split.

CONCEPT: before any machine learning happens, you need labeled data organized
so a program can find it. The convention we use here (and that PyTorch's
`ImageFolder` understands natively) is:

    data/kanto151/train/Bulbasaur/xxx.jpg
    data/kanto151/train/Bulbasaur/yyy.jpg
    data/kanto151/train/Charmander/zzz.jpg
    ...
    data/kanto151/val/Bulbasaur/aaa.jpg
    ...

i.e. one folder per class (Pokemon), and a separate held-out "val" (validation)
split that the model never trains on -- we use it purely to check whether the
model is actually learning to generalize, or just memorizing ("overfitting")
the training images. A typical split is 80-85% train / 15-20% val.

USAGE:
    1. Download a Pokemon image dataset from Kaggle (see README.md for links)
       and unzip it into: data/raw/
       (whatever internal structure it has is fine -- this script searches
       recursively for folders/files that match a Kanto Pokemon name)
    2. Run:  python3 src/prepare_data.py
    3. It will print how many images it found per Pokemon, and write the
       train/val split into data/kanto151/

If a Pokemon has very few images, this script will still work but that class
will be harder for the model to learn -- pay attention to the per-class
counts it prints.
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from kanto_pokemon import KANTO_151, match_kanto_name

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def find_images_by_pokemon(raw_dir: Path) -> dict:
    """
    Walk raw_dir recursively. For every image file, decide which Pokemon it
    belongs to by checking (a) its parent folder name and (b) its own
    filename (minus extension/digits) against the Kanto name list.
    """
    found = defaultdict(list)
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
            continue

        candidates = [path.parent.name, path.stem]
        matched = None
        for candidate in candidates:
            # strip trailing numbers/underscores like "Bulbasaur_01" -> "Bulbasaur"
            cleaned = "".join(ch for ch in candidate if not ch.isdigit()).strip("_- ")
            matched = match_kanto_name(cleaned) or match_kanto_name(candidate)
            if matched:
                break

        if matched:
            found[matched].append(path)

    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", help="Where you unzipped the Kaggle dataset")
    parser.add_argument("--out-dir", default="data/kanto151", help="Where to write the train/val split")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--min-images", type=int, default=4,
                         help="Skip classes with fewer than this many images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        raise SystemExit(
            f"'{raw_dir}' does not exist. Download a dataset (see README.md) "
            f"and unzip it there first."
        )

    random.seed(args.seed)

    found = find_images_by_pokemon(raw_dir)

    missing = [name for name in KANTO_151 if name not in found]
    if missing:
        print(f"Warning: found 0 images for {len(missing)} Kanto Pokemon: {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    total_train, total_val = 0, 0
    kept_classes = 0
    for name in KANTO_151:
        images = found.get(name, [])
        if len(images) < args.min_images:
            if images:
                print(f"Skipping {name}: only {len(images)} image(s), need >= {args.min_images}")
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * args.val_fraction))
        val_images = images[:n_val]
        train_images = images[n_val:]

        for split_name, split_images in [("train", train_images), ("val", val_images)]:
            dest_dir = out_dir / split_name / name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_images:
                dest = dest_dir / f"{src.stem}_{abs(hash(str(src))) % 100000}{src.suffix.lower()}"
                shutil.copy2(src, dest)

        total_train += len(train_images)
        total_val += len(val_images)
        kept_classes += 1
        print(f"{name:15s} train={len(train_images):3d} val={len(val_images):3d}")

    print()
    print(f"Done. {kept_classes}/{len(KANTO_151)} Pokemon classes kept.")
    print(f"Total train images: {total_train}  |  Total val images: {total_val}")
    print(f"Written to: {out_dir}/")


if __name__ == "__main__":
    main()
