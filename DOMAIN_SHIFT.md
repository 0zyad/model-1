# Domain Shift Handling & Deployment Guide

How to adapt the Proppant QC model when the operating environment changes
(new camera, lighting, background, resolution) and how to deploy to edge
devices like NVIDIA Jetson.

---

## 1. What Is Domain Shift?

Your training images were captured with:
- iPhone camera, white background, indoor lighting, ~2268x4032 resolution

Production environments may differ:
- FLIR industrial camera (different sensor characteristics, IR capability)
- Conveyor belt background (textured, moving, non-white)
- LED ring/bar lighting (uniform but different color temperature)
- Different resolution and field of view

**Symptoms of domain shift:**
- Confidence scores drop significantly (e.g., from 0.85 to 0.40)
- Increased false negatives (particles not detected)
- Sand misclassified as proppant (or vice versa)
- mAP drops below 70% on new-environment images

---

## 2. When to Fine-Tune

Fine-tune the model when any of these occur:
- mAP on new-environment images drops below **80%**
- Sand detection precision drops below **85%** (critical safety metric)
- False negative rate exceeds **15%** (particles missed entirely)
- Processing time exceeds **20 seconds** on target hardware

---

## 3. Fine-Tuning Procedure

### Step 1: Collect New Data
- Capture **20-30 images** from the new environment
- Include all particle types (pure 40/70, pure 20/40, mixed, mixed with sand)
- Vary lighting conditions if possible
- Include edge cases (dense packing, partial occlusion)

### Step 2: Annotate
- Follow the same LABELING_GUIDE.md instructions
- Use the same 3 classes: proppant_40_70, proppant_20_40, sand
- Add to the existing dataset (DO NOT remove original training data)

### Step 3: Fine-Tune from Existing Weights

```bash
python train.py \
    --model runs/proppant_seg/weights/best.pt \
    --epochs 50 \
    --batch 8
```

To use a lower learning rate (recommended for fine-tuning), edit
`train.py` and change `lr0=0.01` to `lr0=0.001` before running.

### Step 4: Validate
- Compare mAP before and after fine-tuning
- Check per-class performance (especially sand detection)
- Test on held-out images from BOTH the original and new environments

---

## 4. Avoiding Catastrophic Forgetting

Catastrophic forgetting = the model "forgets" old data when trained only
on new data. Mitigation strategies:

1. **Keep original data in the training set.** Never fine-tune on only
   new-environment images. Combine old + new data.

2. **Lower learning rate.** Use `lr0=0.001` instead of `0.01` for
   fine-tuning. This makes smaller weight updates.

3. **Fewer epochs.** Use 30-50 epochs for fine-tuning, not 150.
   The model already has good features; it just needs adaptation.

4. **Freeze early layers (optional).** If forgetting is severe:
   ```python
   model = YOLO("best.pt")
   # Freeze backbone (first 10 layers)
   for i, (name, param) in enumerate(model.model.named_parameters()):
       if i < 10:
           param.requires_grad = False
   ```

5. **Monitor validation on old data.** Track mAP on original val set
   during fine-tuning. If it drops more than 5%, stop and adjust.

---

## 5. NVIDIA Jetson Deployment

### Export to TensorRT (fastest on Jetson)

```bash
yolo export model=runs/proppant_seg/weights/best.pt \
     format=engine \
     imgsz=640 \
     device=0 \
     half=True
```

This creates a `best.engine` file optimized for the specific Jetson GPU.
Note: The `.engine` file is NOT portable between different Jetson models
(e.g., Orin vs Xavier). Export on the target device.

### Export to ONNX (portable)

```bash
yolo export model=runs/proppant_seg/weights/best.pt \
     format=onnx \
     imgsz=640 \
     simplify=True
```

ONNX models run on any platform with ONNX Runtime. Slower than TensorRT
but portable.

### Using the Exported Model

```python
from ultralytics import YOLO

# TensorRT
model = YOLO("best.engine", task="segment")

# ONNX
model = YOLO("best.onnx", task="segment")

# The rest of the inference code stays the same
results = model.predict(source=image, conf=0.25, iou=0.45)
```

### Model Size Recommendations

| Hardware          | Recommended Model   | Expected FPS |
|-------------------|---------------------|-------------|
| Jetson Orin Nano  | yolov8n-seg (nano)  | 15-30 FPS   |
| Jetson Orin NX    | yolov8s-seg (small) | 20-40 FPS   |
| Desktop GPU       | yolov8m-seg (medium)| 40-80 FPS   |
| CPU only          | yolov8n-seg (nano)  | 2-5 FPS     |

### Jetson Setup Checklist

1. Install JetPack SDK (includes CUDA, cuDNN, TensorRT)
2. Install PyTorch for Jetson (from NVIDIA's wheel):
   ```bash
   pip install torch torchvision  # Use NVIDIA's Jetson-specific wheels
   ```
3. Install ultralytics: `pip install ultralytics`
4. Export model to TensorRT on the Jetson itself
5. Update `config.py`: set `MODEL_BASE` to the `.engine` path
6. Set `PIXELS_PER_MM` after calibrating with the production camera

---

## 6. Camera Transition Checklist

When switching from iPhone to FLIR industrial camera:

- [ ] Capture calibration image with a ruler or reference object
- [ ] Measure `PIXELS_PER_MM` and update `config.py`
- [ ] Capture 20-30 sample images across all particle categories
- [ ] Annotate and add to dataset
- [ ] Fine-tune model (lr0=0.001, 50 epochs)
- [ ] Validate mAP on both old and new images
- [ ] Export to target format (TensorRT for Jetson, ONNX for portability)
- [ ] Run end-to-end test with app.py on target hardware
- [ ] Verify processing time stays under 20 seconds
