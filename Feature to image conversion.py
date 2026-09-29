"""
Feature-to-Image Conversion Pipeline  (Non-Overlapping Windows)
================================================================
For each class separately:
  - Take rows in groups of 16 (non-overlapping)
  - Each group of 16 rows × 16 features → one 16×16 matrix → one image
  - e.g. 39,922 rows ÷ 16 = 2,495 images  (last incomplete group discarded)

Output:
  - 64×64 zero-padded grayscale PNG  (16×16 centered, 24px padding each side)
  - Step-by-step visualization for first STEP_VIS_COUNT samples per class
  - Excel index with full paths, filenames, formats, labels

Author: Generated for Article_MA1
Dataset: Augmented_Dataset_Normalized.xlsx
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_FILE  = r"E:\Articles\Article_24\Article_MA1\Augmented_Dataset_Normalized.xlsx"
SHEET_NAME  = "Augmented_Data_Normalized"
BASE_OUTPUT = r"E:\Articles\Article_24\Article_MA1\Result"

DIR_64_CLASS0    = os.path.join(BASE_OUTPUT, "images_64x64", "class_0_low_interaction")
DIR_64_CLASS1    = os.path.join(BASE_OUTPUT, "images_64x64", "class_1_high_interaction")
DIR_STEPS        = os.path.join(BASE_OUTPUT, "step_by_step_visualization")
EXCEL_INDEX_PATH = os.path.join(BASE_OUTPUT, "image_index.xlsx")

IMAGE_SIZE     = 16   # rows and columns per image window
PADDED_SIZE    = 64   # final padded image size
FEATURE_COLS   = 16   # number of feature columns (excluding label)
STEP_VIS_COUNT = 5    # step-by-step visualizations to save PER CLASS


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def create_directories():
    for d in [DIR_64_CLASS0, DIR_64_CLASS1, DIR_STEPS]:
        os.makedirs(d, exist_ok=True)
    print("[OK] Output directories created.")


def normalize_to_uint8(matrix: np.ndarray) -> np.ndarray:
    """Min-Max normalization of a 2D matrix → uint8 [0, 255]."""
    mn, mx = matrix.min(), matrix.max()
    if mx - mn < 1e-10:
        return np.zeros_like(matrix, dtype=np.uint8)
    return ((matrix - mn) / (mx - mn) * 255.0).astype(np.uint8)


def pad_to_64(img_16: np.ndarray) -> np.ndarray:
    """
    Place a 16×16 image in the centre of a 64×64 zero canvas.
    Padding = (64 - 16) / 2 = 24 pixels on every side.
    """
    canvas = np.zeros((PADDED_SIZE, PADDED_SIZE), dtype=np.uint8)
    pad = (PADDED_SIZE - IMAGE_SIZE) // 2   # 24
    canvas[pad:pad + IMAGE_SIZE, pad:pad + IMAGE_SIZE] = img_16
    return canvas


def save_grayscale_png(array: np.ndarray, filepath: str):
    Image.fromarray(array, mode='L').save(filepath)


# ─────────────────────────────────────────────
# STEP-BY-STEP VISUALIZATION
# ─────────────────────────────────────────────

def save_step_visualization(raw_16: np.ndarray,
                             norm_16: np.ndarray,
                             padded_64: np.ndarray,
                             image_idx: int,
                             label: int,
                             start_row: int,
                             end_row: int,
                             feature_names: list,
                             save_path: str):
    """
    4-panel figure illustrating the full conversion pipeline for one image window:
      Panel 1 – Raw feature values  (16×16 heatmap)
      Panel 2 – Normalized [0, 255] (16×16 grayscale)
      Panel 3 – Zero-padding diagram (64×64, padding = blue)
      Panel 4 – Final 64×64 grayscale image
    + bottom bar chart of flattened pixel values
    """
    class_name = "High Interaction (y=1)" if label == 1 else "Low Interaction (y=0)"

    fig = plt.figure(figsize=(22, 13))
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle(
        (f"Feature → Image Conversion  |  Image #{image_idx}  |  "
         f"Class: {class_name}  |  Source rows: {start_row}–{end_row}"),
        fontsize=13, fontweight='bold', y=0.98, color='#1F3864'
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.42, hspace=0.55)

    # ── Panel 1: Raw feature values ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(raw_16, cmap='RdYlGn', aspect='auto')
    ax1.set_title("Step 1\nRaw Feature Values\n(16 rows × 16 features)",
                  fontsize=10, fontweight='bold', color='#1F3864')
    ax1.set_xlabel("Feature index (column)", fontsize=8)
    ax1.set_ylabel("Sample index (row)", fontsize=8)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    for r in range(IMAGE_SIZE):
        for c in range(IMAGE_SIZE):
            ax1.text(c, r, f"{raw_16[r,c]:.2f}", ha='center', va='center',
                     fontsize=3.5, color='black')

    # ── Panel 2: Normalized 0-255 ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(norm_16, cmap='gray', vmin=0, vmax=255, aspect='auto')
    ax2.set_title("Step 2\nNormalized [0, 255]\n(16×16 grayscale)",
                  fontsize=10, fontweight='bold', color='#1F3864')
    ax2.set_xlabel("Feature index (column)", fontsize=8)
    ax2.set_ylabel("Sample index (row)", fontsize=8)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    for r in range(IMAGE_SIZE):
        for c in range(IMAGE_SIZE):
            ax2.text(c, r, str(norm_16[r, c]), ha='center', va='center',
                     fontsize=3.5,
                     color='white' if norm_16[r, c] < 128 else 'black')

    # ── Panel 3: Padding diagram ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    pad = (PADDED_SIZE - IMAGE_SIZE) // 2   # 24

    pad_rgb = np.zeros((PADDED_SIZE, PADDED_SIZE, 3), dtype=np.uint8)
    pad_rgb[:, :, 2] = 60                                               # blue padding
    for ch in range(3):
        pad_rgb[pad:pad+IMAGE_SIZE, pad:pad+IMAGE_SIZE, ch] = norm_16   # gray centre

    ax3.imshow(pad_rgb, aspect='auto')
    rect = plt.Rectangle((pad-0.5, pad-0.5), IMAGE_SIZE, IMAGE_SIZE,
                          edgecolor='red', facecolor='none', linewidth=2)
    ax3.add_patch(rect)
    ax3.set_title("Step 3\nZero-Padding Added\n(64×64 | blue=padding | red=16×16 region)",
                  fontsize=10, fontweight='bold', color='#1F3864')
    ax3.set_xlabel("Pixel x", fontsize=8)
    ax3.set_ylabel("Pixel y", fontsize=8)
    ax3.annotate('', xy=(pad-0.5, PADDED_SIZE-2), xytext=(0, PADDED_SIZE-2),
                 arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.5))
    ax3.text(pad//2, PADDED_SIZE-5, '24 px', ha='center',
             color='yellow', fontsize=8, fontweight='bold')

    # ── Panel 4: Final 64×64 ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(padded_64, cmap='gray', vmin=0, vmax=255, aspect='auto')
    ax4.set_title("Step 4\nFinal 64×64 Image\n(grayscale PNG)",
                  fontsize=10, fontweight='bold', color='#1F3864')
    ax4.set_xlabel("Pixel x", fontsize=8)
    ax4.set_ylabel("Pixel y", fontsize=8)

    # ── Bottom: flattened pixel values bar chart ──────────────────────────
    ax5 = fig.add_subplot(gs[1, :])
    flat_vals  = raw_16.flatten()
    colors_bar = ['#2196F3' if v >= 0 else '#F44336' for v in flat_vals]
    ax5.bar(range(len(flat_vals)), flat_vals, color=colors_bar,
            edgecolor='none', width=0.85)
    ax5.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax5.set_title(
        "Feature Values — flattened 16×16 = 256 values  |  Blue: positive   Red: negative",
        fontsize=10, fontweight='bold', color='#1F3864'
    )
    ax5.set_xlabel("Pixel index (row-major order)", fontsize=8)
    ax5.set_ylabel("Feature value", fontsize=8)
    ax5.set_xlim(-1, len(flat_vals))

    # Column boundary lines (every 16 pixels = one feature column)
    for i in range(1, IMAGE_SIZE):
        ax5.axvline(i * IMAGE_SIZE - 0.5, color='gray', linewidth=0.4, linestyle=':')

    # Feature name labels on secondary x-axis
    ax5_top = ax5.twiny()
    ax5_top.set_xlim(ax5.get_xlim())
    tick_pos = [i * IMAGE_SIZE + IMAGE_SIZE // 2 for i in range(IMAGE_SIZE)]
    ax5_top.set_xticks(tick_pos)
    ax5_top.set_xticklabels(feature_names, rotation=40, ha='left', fontsize=7)

    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(fig)
    print(f"      [step-vis] {os.path.basename(save_path)}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Feature-to-Image Conversion Pipeline  (Non-Overlapping)")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading data from:\n      {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    print(f"      Total rows: {len(df):,}   Columns: {df.shape[1]}")

    feature_cols = [c for c in df.columns if c != 'y'][:FEATURE_COLS]
    labels       = df['y'].values.astype(int)
    features     = df[feature_cols].values.astype(np.float32)

    n_class0 = (labels == 0).sum()
    n_class1 = (labels == 1).sum()
    img0     = n_class0 // IMAGE_SIZE
    img1     = n_class1 // IMAGE_SIZE

    print(f"\n      Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"\n      Class 0 (Low Interaction) : {n_class0:,} rows  "
          f"→  {n_class0} ÷ {IMAGE_SIZE} = {img0} images  "
          f"(last {n_class0 % IMAGE_SIZE} rows discarded)")
    print(f"      Class 1 (High Interaction): {n_class1:,} rows  "
          f"→  {n_class1} ÷ {IMAGE_SIZE} = {img1} images  "
          f"(last {n_class1 % IMAGE_SIZE} rows discarded)")

    # ── 2. Create directories ─────────────────────────────────────────────
    print("\n[2/5] Creating output directories...")
    create_directories()

    # ── 3. Process each class ─────────────────────────────────────────────
    print(f"\n[3/5] Converting features → 64×64 images "
          f"(step-vis: first {STEP_VIS_COUNT} per class)...")

    idx_class0 = np.where(labels == 0)[0]
    idx_class1 = np.where(labels == 1)[0]
    records    = []

    def process_class(class_indices: np.ndarray, label: int):
        """
        Slide a non-overlapping window of IMAGE_SIZE rows over the class data.
        Each window → one 16×16 matrix → normalize → pad to 64×64 → save PNG.
        """
        class_name = "low_interaction"  if label == 0 else "high_interaction"
        dir_64     = DIR_64_CLASS0      if label == 0 else DIR_64_CLASS1
        label_str  = f"class_{label}_{class_name}"

        # Extract only the feature rows for this class
        class_features = features[class_indices]   # shape: (N, 16)
        n_rows         = len(class_features)
        n_images       = n_rows // IMAGE_SIZE       # complete windows only

        print(f"\n      Processing Class {label} ({class_name}): "
              f"{n_rows:,} rows → {n_images:,} images")

        for img_i in range(n_images):
            start = img_i * IMAGE_SIZE
            end   = start + IMAGE_SIZE

            # ── Build 16×16 matrix: 16 rows × 16 features (NO tiling) ──
            raw_matrix  = class_features[start:end, :]    # (16, 16) — real data
            norm_matrix = normalize_to_uint8(raw_matrix)  # uint8 [0,255]
            padded_64   = pad_to_64(norm_matrix)          # (64, 64)

            fname_64 = f"{label_str}_img_{img_i+1:05d}_64x64.png"
            path_64  = os.path.join(dir_64, fname_64)
            save_grayscale_png(padded_64, path_64)

            # Step-by-step visualization for first STEP_VIS_COUNT images only
            step_vis_name = ""
            step_vis_path = ""
            if img_i < STEP_VIS_COUNT:
                step_vis_name = f"{label_str}_img_{img_i+1:05d}_steps.png"
                step_vis_path = os.path.join(DIR_STEPS, step_vis_name)
                # Map window rows back to original dataset row indices
                orig_start = int(class_indices[start])
                orig_end   = int(class_indices[end - 1])
                save_step_visualization(
                    raw_16       = raw_matrix,
                    norm_16      = norm_matrix,
                    padded_64    = padded_64,
                    image_idx    = img_i + 1,
                    label        = label,
                    start_row    = orig_start,
                    end_row      = orig_end,
                    feature_names= feature_cols,
                    save_path    = step_vis_path
                )

            records.append({
                "image_index"       : img_i + 1,
                "source_row_start"  : int(class_indices[start]),
                "source_row_end"    : int(class_indices[end - 1]),
                "class_label"       : int(label),
                "class_name"        : class_name,
                "image_64x64_name"  : fname_64,
                "image_64x64_format": "PNG",
                "image_64x64_path"  : path_64,
                "step_vis_name"     : step_vis_name,
                "step_vis_path"     : step_vis_path,
            })

            if (img_i + 1) % 500 == 0:
                print(f"        {img_i+1:,} / {n_images:,} images done...")

        print(f"      Class {label}: {n_images:,} images saved  "
              f"(rows used: {n_images * IMAGE_SIZE:,} / {n_rows:,})")

    process_class(idx_class0, label=0)
    process_class(idx_class1, label=1)

    # ── 4. Save Excel index ───────────────────────────────────────────────
    print(f"\n[4/5] Saving Excel index → {EXCEL_INDEX_PATH}")

    wb  = Workbook()
    ws  = wb.active
    ws.title = "Image Index"

    hfill  = PatternFill("solid", fgColor="1F3864")
    hfont  = Font(bold=True, color="FFFFFF", size=10)
    halign = Alignment(horizontal="center", vertical="center", wrap_text=True)
    bdr    = Border(
        left  =Side(style='thin'), right =Side(style='thin'),
        top   =Side(style='thin'), bottom=Side(style='thin')
    )

    headers    = ["Image #", "Source Row Start", "Source Row End",
                  "Class Label", "Class Name",
                  "64×64 Image Name", "Format", "64×64 Full Path",
                  "Step-Vis Name", "Step-Vis Full Path"]
    col_widths = [10, 18, 16, 13, 24, 45, 10, 72, 45, 72]

    for ci, (hdr, w) in enumerate(zip(headers, col_widths), 1):
        cell = ws.cell(row=1, column=ci, value=hdr)
        cell.fill = hfill; cell.font = hfont
        cell.alignment = halign; cell.border = bdr
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 30

    fill0  = PatternFill("solid", fgColor="FCE4D6")
    fill1  = PatternFill("solid", fgColor="E2EFDA")
    dfont  = Font(size=9)
    dalign = Alignment(vertical="center")

    for ri, rec in enumerate(records, 2):
        row_data = [
            rec["image_index"],      rec["source_row_start"], rec["source_row_end"],
            rec["class_label"],      rec["class_name"],
            rec["image_64x64_name"], rec["image_64x64_format"], rec["image_64x64_path"],
            rec["step_vis_name"],    rec["step_vis_path"],
        ]
        fill = fill0 if rec["class_label"] == 0 else fill1
        for ci, val in enumerate(row_data, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.fill = fill; cell.font = dfont
            cell.alignment = dalign; cell.border = bdr

    ws.freeze_panes = "A2"

    # ── Summary sheet ─────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Summary")
    ws2.column_dimensions["A"].width = 40
    ws2.column_dimensions["B"].width = 24
    hfill2 = PatternFill("solid", fgColor="2E5496")

    n0 = sum(1 for r in records if r["class_label"] == 0)
    n1 = sum(1 for r in records if r["class_label"] == 1)
    summary = [
        ("Parameter",                       "Value"),
        ("Total images generated",           len(records)),
        ("Class 0 images (Low Interaction)", n0),
        ("Class 1 images (High Interaction)",n1),
        ("Window size (rows per image)",     IMAGE_SIZE),
        ("Features per row",                 FEATURE_COLS),
        ("Matrix shape per image",           f"{IMAGE_SIZE} × {IMAGE_SIZE}"),
        ("Overlap between windows",          "None (non-overlapping)"),
        ("Original image size",              "16 × 16 pixels"),
        ("Padded image size",                "64 × 64 pixels"),
        ("Padding per side",                 "24 pixels (zero-padding)"),
        ("Image format",                     "PNG (grayscale, uint8)"),
        ("Normalization",                    "Min-Max per image → [0, 255]"),
        ("Step-vis samples per class",       STEP_VIS_COUNT),
        ("Total step-vis files",             STEP_VIS_COUNT * 2),
        ("Feature columns",                  str(feature_cols)),
        ("Source file",                      INPUT_FILE),
        ("Output directory",                 BASE_OUTPUT),
    ]
    for ri, (k, v) in enumerate(summary, 1):
        c1 = ws2.cell(row=ri, column=1, value=k)
        c2 = ws2.cell(row=ri, column=2, value=str(v))
        if ri == 1:
            for c in [c1, c2]:
                c.fill = hfill2
                c.font = Font(bold=True, color="FFFFFF", size=10)
        c1.border = bdr; c2.border = bdr

    wb.save(EXCEL_INDEX_PATH)

    # ── 5. Final report ───────────────────────────────────────────────────
    print("\n[5/5] Pipeline complete!")
    print("=" * 65)
    print(f"  64×64 images  class 0 : {DIR_64_CLASS0}")
    print(f"  64×64 images  class 1 : {DIR_64_CLASS1}")
    print(f"  Step visuals          : {DIR_STEPS}  ({STEP_VIS_COUNT*2} files)")
    print(f"  Excel index           : {EXCEL_INDEX_PATH}")
    print("=" * 65)
    print(f"  Total images generated : {len(records):,}  "
          f"(class 0: {n0:,}  |  class 1: {n1:,})")
    print(f"  Step visualizations    : {STEP_VIS_COUNT*2}  "
          f"({STEP_VIS_COUNT} per class)")
    print("=" * 65)


if __name__ == "__main__":
    main()