import os
import glob
import cv2
import yaml
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from multiprocessing import freeze_support

# =========================
# CONFIG
# =========================
RUN_DIR = r"D:\NCKH\Forest\forest_seg"
BEST_PT = os.path.join(RUN_DIR, "weights", "best.pt")
RESULTS_CSV = os.path.join(RUN_DIR, "results.csv")

DATA_YAML = r"D:\NCKH\Forest\data.yaml"
TEST_IMG_DIR = r"D:\NCKH\Forest\test\images"

OUT_DIR = r"D:\NCKH\Forest\outputs"
PRED_DIR = os.path.join(OUT_DIR, "pred_green")
COMPARE_DIR = os.path.join(OUT_DIR, "compare_lr")
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)

CONF_THRES = 0.60
MASK_THRES = 0.50


def is_img(p):
    return p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    names = y.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    return list(names)


def pick_tree_class_id(names_list):
    if len(names_list) == 1:
        return 0

    keywords = ["tree", "forest", "vegetation", "wood", "canopy"]
    for i, n in enumerate([x.lower() for x in names_list]):
        if any(k in n for k in keywords):
            return i

    print("⚠️ Không tìm thấy class cây → mặc định class 0")
    return 0


def overlay_green_mask(bgr_img, mask_hw, alpha=0.45):
    h, w = bgr_img.shape[:2]
    if mask_hw.shape != (h, w):
        mask_hw = cv2.resize(
            mask_hw.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    out = bgr_img.copy()
    green = np.zeros_like(out)
    green[:, :, 1] = 255
    blended = cv2.addWeighted(out, 1 - alpha, green, alpha, 0)
    out[mask_hw] = blended[mask_hw]
    return out


def print_metrics(model):
    print("\n================= METRICS =================")

    # -------- LOSS --------
    if os.path.isfile(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        last = df.iloc[-1]

        if "val/seg_loss" in df.columns:
            print(f"1) VAL SEG LOSS = {last['val/seg_loss']:.6f}")
        else:
            print("1) VAL SEG LOSS: không có")
    else:
        print("1) results.csv không tồn tại")

    # -------- mAP --------
    print("\n2) mAP METRICS:")

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0
    )

    if hasattr(metrics, "seg") and metrics.seg is not None:
        print(f"   mAP@0.5      = {metrics.seg.map50:.4f}")
        print(f"   mAP@0.5:0.95 = {metrics.seg.map:.4f}")
    else:
        print("   ❌ Model KHÔNG PHẢI segmentation")


def main():
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    print("DEVICE:", DEVICE)

    assert os.path.isfile(DATA_YAML)
    assert os.path.isdir(TEST_IMG_DIR)
    assert os.path.isfile(BEST_PT)

    # -------- LOAD MODEL (CHỈ 1 LẦN) --------
    model = YOLO(BEST_PT)

    names_list = load_class_names(DATA_YAML)
    TREE_CLASS_ID = pick_tree_class_id(names_list)

    print("CLASSES:", names_list)
    print("TREE_CLASS_ID:", TREE_CLASS_ID)
    print(f"CONF={CONF_THRES} | MASK={MASK_THRES}")

    # -------- METRICS --------
    print_metrics(model)

    # -------- PREDICT --------
    img_paths = [p for p in sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*"))) if is_img(p)]
    print(f"\nFound {len(img_paths)} test images")

    for img_path in img_paths:
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue

        results = model.predict(
            source=img_path,
            device=DEVICE,
            conf=CONF_THRES,
            iou=0.50,
            verbose=False
        )

        r = results[0]
        pred_img = bgr.copy()

        if r.masks is not None and r.boxes is not None:
            masks = r.masks.data.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()

            keep = np.where((cls == TREE_CLASS_ID) & (conf >= CONF_THRES))[0]
            if len(keep) > 0:
                combined = (masks[keep].max(axis=0) > MASK_THRES)
                pred_img = overlay_green_mask(bgr, combined)

        base = os.path.basename(img_path)
        cv2.imwrite(os.path.join(PRED_DIR, base), pred_img)

        compare = np.concatenate([bgr, pred_img], axis=1)
        cv2.imwrite(os.path.join(COMPARE_DIR, base), compare)

    print("\n✅ DONE")
    print("Pred:", PRED_DIR)
    print("Compare:", COMPARE_DIR)


if __name__ == "__main__":
    freeze_support()
    main()
