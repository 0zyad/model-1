# Proppant QC — Image Annotation Guide

This guide explains how to annotate images for training the YOLOv8-seg
instance segmentation model.

---

## 1. Tool Selection

**Recommended:** [Roboflow](https://roboflow.com) (free tier)
- Browser-based, no installation
- SAM-assisted "Smart Polygon" auto-annotation
- Direct export to YOLO Segmentation format

**Alternative:** [CVAT](https://www.cvat.ai)
- Open-source, self-hostable
- Good polygon tools
- Export as YOLO segmentation

---

## 2. Project Setup

1. Create a new project in Roboflow
2. Set project type: **Instance Segmentation**
3. Create exactly **3 classes**:
   - `proppant_40_70`
   - `proppant_20_40`
   - `sand`
4. Upload all images from `dataset/images/train/` and `dataset/images/val/`

---

## 3. Annotation Rules by Image Category

### P4070_xxx.jpg (40/70 proppant only)
- **Every particle** → `proppant_40_70`
- Small, dark, uniformly spherical particles on white background
- All particles are the same type — no size discrimination needed

### P2040_xxx.jpg (20/40 proppant only)
- **Every particle** → `proppant_20_40`
- Larger dark spherical particles on white background
- Noticeably bigger than 40/70 particles

### MIX2_xxx.jpg (mixed 40/70 + 20/40)
- Contains **two sizes** of dark particles
- **Large particles** → `proppant_20_40`
- **Small particles** → `proppant_40_70`
- Use pure-sample images (P4070, P2040) as size reference
- When in doubt, compare to the median size in each group

### MIXS_xxx.jpg (mixed with sand)
- Contains **three types**:
  - **Dark, opaque, large spheres** → `proppant_20_40`
  - **Dark, opaque, small spheres** → `proppant_40_70`
  - **Light, translucent, amber/yellowish irregular grains** → `sand`
- Sand is visually distinct by color (lighter) and shape (irregular)

---

## 4. Polygon Annotation Technique

1. Use the **Smart Polygon** or **SAM auto-segment** tool in Roboflow
2. Click on each particle — SAM will suggest a mask boundary
3. Adjust the polygon if needed to tightly trace the particle edge
4. Each particle gets its **own individual mask** (do not merge clusters)

### Quality rules:
- Minimum **6 vertices** per polygon for round particles
- **Skip** particles that are:
  - Heavily occluded (less than 50% visible)
  - Cut off at the image edge
  - Too blurry to classify
- For touching particles: trace each one individually

### How many particles to annotate per image:
- **Ideal:** All clearly visible particles (could be 100-400+ per image)
- **Minimum:** At least 50 particles per image
- **Priority:** Focus on clearly separated, well-focused particles first
- SAM-assisted tools can greatly speed this up

---

## 5. Export Settings

1. In Roboflow, go to **Generate** → **Export**
2. Format: **YOLOv8** (select "Instance Segmentation" variant)
3. This produces `.txt` files where each line is:
   ```
   class_id x1 y1 x2 y2 x3 y3 ... xn yn
   ```
   All coordinates are normalized (0.0–1.0)

4. Class IDs:
   - `0` = proppant_40_70
   - `1` = proppant_20_40
   - `2` = sand

---

## 6. File Placement

After exporting from Roboflow:

1. Copy label `.txt` files for training images into:
   ```
   dataset/labels/train/
   ```

2. Copy label `.txt` files for validation images into:
   ```
   dataset/labels/val/
   ```

3. Each label file must match its image filename:
   - `P4070_001.jpg` → `P4070_001.txt`
   - `MIX2_015.jpg` → `MIX2_015.txt`

---

## 7. Verification

After placing label files, verify the dataset:

```bash
# Check that every train image has a label
ls dataset/images/train/ | wc -l
ls dataset/labels/train/ | wc -l
# These counts should match

# Inspect a label file
head -5 dataset/labels/train/P4070_001.txt
# Should show lines like: 0 0.234 0.567 0.245 0.578 ...
```

---

## 8. Tips for Faster Annotation

- **Start with pure samples** (P4070, P2040) — single class, faster labeling
- **Use Roboflow's auto-label** feature to pre-annotate, then manually correct
- **Annotate in batches** of 5-10 images, export, check quality, repeat
- **Consistency matters more than quantity** — 40 well-annotated images
  are better than 76 poorly annotated ones
