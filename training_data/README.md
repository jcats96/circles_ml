# Training Data

This folder holds the PNG images and their annotations used to train the
circle-counting model.

---

## Folder Layout

```
training_data/
├── README.md          ← this file
├── labels.csv         ← annotation file (one row per image)
└── images/
    ├── img_001.png
    ├── img_002.png
    └── ...
```

Place all of your 32×32 PNG images inside the `images/` sub-folder.

---

## Annotation Format

Create a file called `labels.csv` in this folder with two columns:

```
filename,circles
img_001.png,0
img_002.png,1
img_003.png,3
```

| Column | Description |
|--------|-------------|
| `filename` | The PNG file name (just the base name, no path prefix) |
| `circles` | The number of circles in that image (integer ≥ 0) |

The file must have a header row exactly as shown above so that the training
script can read it reliably.

---

## Approaches to Creating Annotations

### Option A – Manual Annotation (small datasets)

1. Open each PNG in any image viewer.
2. Count the circles by eye.
3. Add a row to `labels.csv` with the file name and count.

Free tools that can help if you want a lightweight GUI:
- **LabelImg** (`pip install labelImg`) – originally for bounding boxes but
  you can record counts in a separate spreadsheet.
- A simple spreadsheet (Excel, LibreOffice Calc, Google Sheets) — save as CSV
  when done.

### Option B – Programmatic Generation (recommended for larger datasets)

Because the images are synthetic (drawn circles on a blank canvas) you can
generate images **and** their labels at the same time:

```python
import os, random, csv
from PIL import Image, ImageDraw

os.makedirs("training_data/images", exist_ok=True)

with open("training_data/labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "circles"])

    for i in range(1000):           # generate 1 000 images
        img = Image.new("L", (32, 32), color=0)   # black background
        draw = ImageDraw.Draw(img)

        n = random.randint(0, 5)    # 0 – 5 circles per image
        for _ in range(n):
            r = random.randint(2, 6)
            x = random.randint(r, 31 - r)
            y = random.randint(r, 31 - r)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=255)

        name = f"img_{i:05d}.png"
        img.save(f"training_data/images/{name}")
        writer.writerow([name, n])
```

Run `pip install Pillow` first if Pillow is not already installed.

### Option C – Semi-Automated Annotation

If you already have images but want to avoid manual counting:

1. Use a classical computer-vision method (e.g. Hough Circle Transform in
   OpenCV) to get an approximate count:

   ```python
   import cv2
   import numpy as np

   img = cv2.imread("training_data/images/img_001.png", cv2.IMREAD_GRAYSCALE)
   circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=5,
                               param1=50, param2=15, minRadius=2, maxRadius=8)
   count = 0 if circles is None else len(circles[0])
   print(count)
   ```

2. Review the count and correct any errors before adding to `labels.csv`.

---

## Loading the Dataset in Python

```python
import os
import csv
import numpy as np
from PIL import Image

def load_dataset(data_dir="training_data"):
    images, labels = [], []
    csv_path = os.path.join(data_dir, "labels.csv")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(data_dir, "images", row["filename"])
            img = Image.open(img_path).convert("L")       # grayscale
            arr = np.array(img, dtype=np.float32) / 255.0 # normalize to [0,1]
            images.append(arr.reshape(32, 32, 1))
            labels.append(int(row["circles"]))
    return np.array(images), np.array(labels, dtype=np.float32)
```

---

## Notes

- Keep images at exactly **32×32 pixels**. If yours are a different size,
  resize them before saving:
  ```python
  img = img.resize((32, 32), Image.NEAREST)
  ```
  `Image.NEAREST` preserves the hard black/white edges better than
  interpolation filters.
- Binary images (pure black/white) should be saved as 8-bit grayscale PNGs.
  Avoid JPEG compression — it introduces grey artifacts on sharp edges.
