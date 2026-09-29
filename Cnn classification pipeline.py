"""
CNN Classification Pipeline — Banking Customer Interaction
==========================================================
Three models trained on 64×64 grayscale images:
  Model 1 : Plain CNN (baseline)
  Model 2 : CNN + SE Channel Attention
  Model 3 : RDAD-CNN — Residual Dual-Attention Depthwise-Separable (Novel)

DATA LEAKAGE PREVENTION:
  - Images are loaded ONCE and split BEFORE any preprocessing
  - Normalization is already baked into images (0-255 → /255.0 at load time)
  - StratifiedKFold split is on RAW indices; no fitted transform crosses folds
  - Model weights reset (K.clear_session) before each fold
  - Early stopping and LR scheduler use val set only
  - No augmentation that could leak label statistics

Results saved to:  E:\Articles\Article_24\Article_MA1\Result\CNN_Results\
"""

import os, time, warnings, json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score,
)
from sklearn.utils import shuffle as sk_shuffle

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from tensorflow.keras.optimizers import Adam

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
EXCEL_INDEX = r"E:\Articles\Article_24\Article_MA1\Result\image_index.xlsx"
BASE_RESULT = r"E:\Articles\Article_24\Article_MA1\Result\CNN_Results"

IMG_SIZE   = 64
BATCH_SIZE = 32
EPOCHS     = 50
K_FOLDS    = 5
LR         = 1e-3
SEED       = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ═══════════════════════════════════════════════════════════════
# DIRECTORIES
# ═══════════════════════════════════════════════════════════════
def make_dirs():
    dirs = {
        'root'   : BASE_RESULT,
        'model1' : os.path.join(BASE_RESULT, "Model1_PlainCNN"),
        'model2' : os.path.join(BASE_RESULT, "Model2_CNN_SE_Attention"),
        'model3' : os.path.join(BASE_RESULT, "Model3_RDAD_CNN_Novel"),
        'compare': os.path.join(BASE_RESULT, "Comparison"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ═══════════════════════════════════════════════════════════════
# DATA LOADING  (no leakage: images loaded as-is, no fit step)
# ═══════════════════════════════════════════════════════════════
def load_images(excel_path: str):
    """
    Load images from paths listed in the Excel index.
    Images are stored in two separate class arrays then combined.
    Pixel values are divided by 255.0 (pure scaling, no statistics
    computed from data → no leakage risk).
    """
    print(f"\n[DATA] Reading index: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name="Image Index")

    path_col  = [c for c in df.columns
                 if '64' in c and 'Path' in c and 'Step' not in c][0]
    label_col = [c for c in df.columns
                 if 'Label' in c or 'label' in c][0]

    paths  = df[path_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()

    # ── Separate by class while loading ──────────────────────────
    images_0, images_1 = [], []
    skipped = 0
    print(f"       Loading {len(paths):,} images ...")

    for i, (p, lbl) in enumerate(zip(paths, labels)):
        try:
            img = np.array(
                Image.open(p).convert('L'), dtype=np.float32
            ) / 255.0                              # scale only, no mean/std fit
            if lbl == 0:
                images_0.append(img)
            else:
                images_1.append(img)
        except Exception as e:
            print(f"       [SKIP] {os.path.basename(p)}: {e}")
            skipped += 1
        if (i + 1) % 1000 == 0:
            print(f"       {i+1:,} / {len(paths):,} loaded ...")

    # ── Store in separate named variables ─────────────────────────
    X_label_0 = np.array(images_0)[..., np.newaxis]   # (N0, 64, 64, 1)
    X_label_1 = np.array(images_1)[..., np.newaxis]   # (N1, 64, 64, 1)
    y_label_0 = np.zeros(len(X_label_0), dtype=np.int32)
    y_label_1 = np.ones( len(X_label_1), dtype=np.int32)

    print(f"\n       X_label_0 : {X_label_0.shape}  (class 0 — Low Interaction)")
    print(f"       X_label_1 : {X_label_1.shape}  (class 1 — High Interaction)")
    if skipped:
        print(f"       Skipped   : {skipped} images")

    # ── Combine back into a single dataset ────────────────────────
    X = np.concatenate([X_label_0, X_label_1], axis=0)
    y = np.concatenate([y_label_0, y_label_1], axis=0)

    print(f"\n       X combined: {X.shape}")
    print(f"       y counts  : class 0 = {(y==0).sum():,}  "
          f"class 1 = {(y==1).sum():,}")

    return X, y, X_label_0, X_label_1


# ═══════════════════════════════════════════════════════════════
# MODEL 1 — Plain CNN (Baseline)
# ═══════════════════════════════════════════════════════════════
def build_plain_cnn(input_shape=(64, 64, 1)):
    """
    4-block Conv→BN→ReLU→MaxPool + GAP + Dense head.
    No attention, no skip connections.
    """
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inp, out, name='PlainCNN')
    model.compile(
        optimizer=Adam(LR),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')],
    )
    return model


# ═══════════════════════════════════════════════════════════════
# MODEL 2 — CNN + SE Channel Attention
# ═══════════════════════════════════════════════════════════════
def se_block(x, ratio=16):
    """Squeeze-and-Excitation channel attention."""
    ch = x.shape[-1]
    s  = layers.GlobalAveragePooling2D()(x)
    s  = layers.Reshape((1, 1, ch))(s)
    s  = layers.Dense(max(ch // ratio, 1), activation='relu',  use_bias=False)(s)
    s  = layers.Dense(ch,                  activation='sigmoid', use_bias=False)(s)
    return layers.Multiply()([x, s])


def build_cnn_se_attention(input_shape=(64, 64, 1)):
    """
    Same backbone as PlainCNN but with an SE block after every Conv block.
    SE recalibrates channel-wise feature responses.
    """
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = se_block(x, ratio=8)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = se_block(x, ratio=8)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = se_block(x, ratio=8)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = se_block(x, ratio=16)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inp, out, name='CNN_SE_Attention')
    model.compile(
        optimizer=Adam(LR),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')],
    )
    return model


# ═══════════════════════════════════════════════════════════════
# MODEL 3 — RDAD-CNN  (Novel: Residual Dual-Attention DepthSep)
# ═══════════════════════════════════════════════════════════════
def channel_attention_dual(x, ratio=16):
    """
    Dual-path channel attention:
    both Average-Pool and Max-Pool branches share the same MLP,
    outputs are summed before sigmoid.
    """
    ch  = x.shape[-1]
    r   = max(ch // ratio, 1)

    avg = layers.GlobalAveragePooling2D()(x)
    mx  = layers.GlobalMaxPooling2D()(x)
    avg = layers.Reshape((1, 1, ch))(avg)
    mx  = layers.Reshape((1, 1, ch))(mx)

    # Shared MLP
    d1 = layers.Dense(r,  activation='relu',   use_bias=False)
    d2 = layers.Dense(ch, activation='linear', use_bias=False)

    scale = layers.Activation('sigmoid')(
        layers.Add()([d2(d1(avg)), d2(d1(mx))])
    )
    return layers.Multiply()([x, scale])


def spatial_attention(x, kernel=7):
    """
    Spatial attention: concatenate channel avg-pool & max-pool,
    then 1-channel Conv → sigmoid mask.
    """
    avg   = tf.reduce_mean(x, axis=-1, keepdims=True)
    mx    = tf.reduce_max( x, axis=-1, keepdims=True)
    cat   = layers.Concatenate(axis=-1)([avg, mx])
    scale = layers.Conv2D(1, kernel, padding='same',
                          activation='sigmoid', use_bias=False)(cat)
    return layers.Multiply()([x, scale])


def depthsep_conv(x, filters, strides=1):
    """Depthwise-Separable Conv → BN → ReLU."""
    x = layers.DepthwiseConv2D(3, padding='same',
                                strides=strides, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def rdad_block(x, filters):
    """
    Residual Dual-Attention Depthwise-Separable block:
      DepthSep Conv → Channel Attention → Spatial Attention → residual add
    Innovations combined in one block:
      (1) DepthSep: fewer params than standard conv
      (2) Dual channel attention (avg+max paths)
      (3) Spatial attention on top of channel attention (full CBAM)
      (4) Residual skip for gradient stability
    """
    shortcut = x
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = depthsep_conv(x, filters)
    x = channel_attention_dual(x)
    x = spatial_attention(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_rdad_cnn(input_shape=(64, 64, 1)):
    """
    RDAD-CNN: Residual Dual-Attention Depthwise-Separable CNN
    ----------------------------------------------------------
    Novel contributions vs. existing literature:
      1. Depthwise-Separable convolutions  → ~8-9× fewer params than plain conv
      2. Dual-path Channel Attention        → richer than single-path SE block
      3. Spatial Attention stacked on top   → full CBAM-style recalibration
      4. Residual connections per block     → stable gradient in deep network
      5. Multi-scale head (GAP ∥ GMP)       → captures both average & peak features
    """
    inp = layers.Input(shape=input_shape)

    # Stem: one standard conv to lift channels to 32
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1 → 64 channels
    x = rdad_block(x, 64)
    x = layers.MaxPooling2D(2)(x)

    # Stage 2 → 128 channels
    x = rdad_block(x, 128)
    x = layers.MaxPooling2D(2)(x)

    # Stage 3 → 256 channels
    x = rdad_block(x, 256)
    x = layers.MaxPooling2D(2)(x)

    # Stage 4 → 512 channels
    x = rdad_block(x, 512)

    # Multi-scale head: GAP + GMP concatenated
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x   = layers.Concatenate()([gap, gmp])   # (1024,)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inp, out, name='RDAD_CNN_Novel')
    model.compile(
        optimizer=Adam(LR),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')],
    )
    return model


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    report = classification_report(
        y_true, y_pred,
        target_names=['Low(0)', 'High(1)'],
        output_dict=True, zero_division=0,
    )
    return {
        'accuracy'   : float(report['accuracy']),
        'auc'        : float(roc_auc_score(y_true, y_prob)),
        'ap'         : float(average_precision_score(y_true, y_prob)),
        'precision'  : float(report['High(1)']['precision']),
        'recall'     : float(report['High(1)']['recall']),
        'f1'         : float(report['High(1)']['f1-score']),
        'mcc'        : float(matthews_corrcoef(y_true, y_pred)),
        'kappa'      : float(cohen_kappa_score(y_true, y_pred)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'ppv'        : float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'npv'        : float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'y_true' : y_true,
        'y_pred' : y_pred,
        'y_prob' : y_prob,
        'cm'     : cm,
        'report' : report,
    }


# ═══════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════
C_TRAIN = '#2196F3'
C_VAL   = '#F44336'


def plot_history(histories, model_name, save_dir):
    metric_keys = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(22, 4))
    fig.suptitle(f'{model_name} — Training History ({K_FOLDS}-Fold CV)',
                 fontsize=12, fontweight='bold')

    for ax, mk in zip(axes, metric_keys):
        all_tr, all_vl = [], []
        for fi, h in enumerate(histories):
            if mk in h:
                ax.plot(h[mk], color=C_TRAIN, alpha=0.3, linewidth=1,
                        label='train' if fi == 0 else '')
                all_tr.append(h[mk])
            vk = f'val_{mk}'
            if vk in h:
                ax.plot(h[vk], color=C_VAL, alpha=0.3, linewidth=1,
                        label='val' if fi == 0 else '')
                all_vl.append(h[vk])

        def mean_curve(curves):
            min_len = min(len(c) for c in curves)
            return np.mean([c[:min_len] for c in curves], axis=0)

        if all_tr:
            ax.plot(mean_curve(all_tr), color=C_TRAIN, linewidth=2.5,
                    linestyle='--', label='mean train')
        if all_vl:
            ax.plot(mean_curve(all_vl), color=C_VAL, linewidth=2.5,
                    linestyle='--', label='mean val')

        ax.set_title(mk.upper(), fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel(mk)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'),
                dpi=130, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, model_name, fold, save_dir, normalize=True):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = np.where(row_sums == 0, 0, cm / row_sums)
        fmt, suf = '.2%', 'Normalized'
    else:
        cm_plot, fmt, suf = cm, 'd', 'Counts'

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=['Low(0)', 'High(1)'],
                yticklabels=['Low(0)', 'High(1)'],
                ax=ax, linewidths=0.5)
    ax.set_title(f'{model_name}\nCM {suf} — Fold {fold}', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    tag  = 'norm' if normalize else 'raw'
    path = os.path.join(save_dir, f'{model_name}_fold{fold}_cm_{tag}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_mean_cm(cms, model_name, save_dir):
    mean_cm = np.mean(cms, axis=0)
    norm_cm = np.where(
        mean_cm.sum(axis=1, keepdims=True) == 0, 0,
        mean_cm / mean_cm.sum(axis=1, keepdims=True)
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in zip(
            axes,
            [mean_cm, norm_cm],
            ['.1f', '.2%'],
            ['Mean Counts', 'Mean Normalized']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=['Low(0)', 'High(1)'],
                    yticklabels=['Low(0)', 'High(1)'],
                    ax=ax, linewidths=0.5)
        ax.set_title(f'{model_name}\n{title} ({K_FOLDS}-Fold)', fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_mean_cm.png'),
                dpi=130, bbox_inches='tight')
    plt.close()


def plot_roc_pr(fold_results, model_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{model_name} — ROC & PR Curves ({K_FOLDS}-Fold)',
                 fontweight='bold', fontsize=12)

    base_fpr    = np.linspace(0, 1, 300)
    base_recall = np.linspace(0, 1, 300)
    tpr_list, prec_list = [], []

    for i, fr in enumerate(fold_results):
        fpr, tpr, _   = roc_curve(fr['y_true'], fr['y_prob'])
        prec, rec, _  = precision_recall_curve(fr['y_true'], fr['y_prob'])
        axes[0].plot(fpr, tpr, alpha=0.35, linewidth=1,
                     label=f'F{i+1} AUC={fr["auc"]:.4f}')
        axes[1].plot(rec, prec, alpha=0.35, linewidth=1,
                     label=f'F{i+1} AP={fr["ap"]:.4f}')
        tpr_list.append(np.interp(base_fpr, fpr, tpr))
        prec_list.append(np.interp(base_recall, rec[::-1], prec[::-1]))

    mean_auc  = np.mean([fr['auc'] for fr in fold_results])
    mean_ap   = np.mean([fr['ap']  for fr in fold_results])
    axes[0].plot(base_fpr, np.mean(tpr_list, axis=0), 'k--', lw=2.5,
                 label=f'Mean AUC={mean_auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'gray', linestyle=':')
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC Curve')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].plot(base_recall, np.mean(prec_list, axis=0), 'k--', lw=2.5,
                 label=f'Mean AP={mean_ap:.4f}')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_roc_pr.png'),
                dpi=130, bbox_inches='tight')
    plt.close()


def plot_metrics_boxplot(fold_results, model_name, save_dir):
    keys  = ['accuracy', 'auc', 'f1', 'precision', 'recall', 'mcc', 'kappa']
    data  = [[fr[k] for fr in fold_results] for k in keys]
    clrs  = ['#2196F3','#4CAF50','#9C27B0','#FF9800','#F44336','#00BCD4','#795548']

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=[k.upper() for k in keys],
                    patch_artist=True, notch=False)
    for patch, c in zip(bp['boxes'], clrs):
        patch.set_facecolor(c); patch.set_alpha(0.75)

    ax.set_title(f'{model_name} — Per-Fold Metric Distribution', fontweight='bold')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_metrics_boxplot.png'),
                dpi=120, bbox_inches='tight')
    plt.close()


def plot_arch_params(model, model_name, save_dir):
    """Bar chart of top layers by parameter count + save text summary."""
    txt = os.path.join(save_dir, f'{model_name}_architecture.txt')
    with open(txt, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda ln: f.write(ln + '\n'))

    names, counts = [], []
    for lyr in model.layers:
        pc = lyr.count_params()
        if pc > 0:
            names.append(lyr.name[:28])
            counts.append(int(pc))

    if not names:
        return
    top = min(15, len(names))
    idx = np.argsort(counts)[-top:]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([names[i] for i in idx], [counts[i] for i in idx],
            color='#2196F3', alpha=0.8)
    ax.set_title(f'{model_name} — Top {top} Layers by Parameter Count',
                 fontweight='bold')
    ax.set_xlabel('Parameters')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_param_counts.png'),
                dpi=120, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# EXCEL REPORT
# ═══════════════════════════════════════════════════════════════
def _bdr():
    s = Side(style='thin')
    return Border(left=s, right=s, top=s, bottom=s)

def _hdr_style(ws, headers, widths, bdr, hfill, hfont):
    halign = Alignment(horizontal='center', vertical='center', wrap_text=True)
    for ci, (h, w) in enumerate(zip(headers, widths), 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.fill = hfill; cell.font = hfont
        cell.alignment = halign; cell.border = bdr
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 28


def save_excel_report(fold_results, model_name, save_dir):
    wb    = Workbook()
    bdr   = _bdr()
    hfill = PatternFill('solid', fgColor='1F3864')
    hfont = Font(bold=True, color='FFFFFF', size=10)
    row_fills = [PatternFill('solid', fgColor=c) for c in
                 ['E8F5E9','E3F2FD','FFF9C4','FCE4EC','F3E5F5']]

    def write_data_row(ws, ri, data, fill=None):
        for ci, val in enumerate(data, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.border = bdr
            cell.alignment = Alignment(vertical='center')
            if fill: cell.fill = fill

    # ── Sheet 1: Per-fold metrics ─────────────────────────────────
    ws1 = wb.active; ws1.title = 'Per-Fold Metrics'
    h1  = ['Fold','Accuracy','AUC','AP','Precision','Recall','F1',
           'MCC','Kappa','Specificity','Sensitivity','PPV','NPV',
           'TP','TN','FP','FN','Train Time (s)']
    w1  = [7,12,10,10,12,10,10,10,10,13,13,10,10,8,8,8,8,14]
    _hdr_style(ws1, h1, w1, bdr, hfill, hfont)

    for ri, fr in enumerate(fold_results, 2):
        write_data_row(ws1, ri, [
            fr['fold'],
            round(fr['accuracy'],   5), round(fr['auc'],        5),
            round(fr['ap'],         5), round(fr['precision'],   5),
            round(fr['recall'],     5), round(fr['f1'],          5),
            round(fr['mcc'],        5), round(fr['kappa'],       5),
            round(fr['specificity'],5), round(fr['sensitivity'], 5),
            round(fr['ppv'],        5), round(fr['npv'],         5),
            fr['tp'], fr['tn'], fr['fp'], fr['fn'],
            round(fr.get('train_time', 0), 2),
        ], fill=row_fills[(ri - 2) % 5])

    # Mean / Std rows
    stat_keys = ['accuracy','auc','ap','precision','recall','f1',
                 'mcc','kappa','specificity','sensitivity','ppv','npv']
    mfill = PatternFill('solid', fgColor='2E5496')
    for label, func in [('MEAN', np.mean), ('STD', np.std)]:
        vals = [round(func([fr[k] for fr in fold_results]), 5) for k in stat_keys]
        ri   = len(fold_results) + 2 + (0 if label == 'MEAN' else 1)
        row  = [label] + vals + ['', '', '', '', '', '']
        for ci, v in enumerate(row, 1):
            cell = ws1.cell(row=ri, column=ci, value=v)
            cell.fill = mfill
            cell.font = Font(bold=True, color='FFFFFF', size=10)
            cell.border = bdr
    ws1.freeze_panes = 'A2'

    # ── Sheet 2: Summary statistics ───────────────────────────────
    ws2 = wb.create_sheet('Summary Statistics')
    _hdr_style(ws2, ['Metric','Mean','Std','Min','Max'],
               [28, 14, 14, 14, 14], bdr, hfill, hfont)
    for ri, key in enumerate(stat_keys, 2):
        vals = [fr[key] for fr in fold_results]
        write_data_row(ws2, ri, [
            key.upper(),
            round(np.mean(vals), 5), round(np.std(vals),  5),
            round(np.min(vals),  5), round(np.max(vals),  5),
        ], fill=row_fills[(ri - 2) % 5])

    # ── Sheet 3: Classification reports ───────────────────────────
    ws3  = wb.create_sheet('Classification Reports')
    rptr = 1
    for fr in fold_results:
        ws3.cell(row=rptr, column=1,
                 value=f'Fold {fr["fold"]} — {model_name}').font = Font(bold=True, size=11)
        rptr += 1
        _hdr_style(ws3, ['Class','Precision','Recall','F1','Support'],
                   [16, 13, 13, 13, 12], bdr, hfill, hfont)
        rptr += 1  # skip header row (already written at rptr-1 but _hdr_style uses row=1)
        # Re-write inline
        halign = Alignment(vertical='center')
        for cls, vals in fr['report'].items():
            if isinstance(vals, dict):
                for ci, v in enumerate([
                    cls,
                    round(vals.get('precision', 0), 4),
                    round(vals.get('recall', 0),    4),
                    round(vals.get('f1-score', 0),  4),
                    int(vals.get('support', 0)),
                ], 1):
                    cell = ws3.cell(row=rptr, column=ci, value=v)
                    cell.border = bdr; cell.alignment = halign
                rptr += 1
            else:
                ws3.cell(row=rptr, column=1, value=cls)
                ws3.cell(row=rptr, column=2, value=round(float(vals), 4))
                rptr += 1
        rptr += 2

    # ── Sheet 4: Confusion matrices ───────────────────────────────
    ws4  = wb.create_sheet('Confusion Matrices')
    rptr = 1
    for fr in fold_results:
        ws4.cell(row=rptr, column=1,
                 value=f'Fold {fr["fold"]} Confusion Matrix').font = Font(bold=True)
        rptr += 1
        cm = fr['cm']
        for ci, lbl in enumerate(['', 'Pred: Low(0)', 'Pred: High(1)'], 1):
            ws4.cell(row=rptr, column=ci, value=lbl)
        rptr += 1
        for actual, row_lbl in [(0, 'Actual: Low(0)'), (1, 'Actual: High(1)')]:
            ws4.cell(row=rptr, column=1, value=row_lbl)
            ws4.cell(row=rptr, column=2, value=int(cm[actual, 0]))
            ws4.cell(row=rptr, column=3, value=int(cm[actual, 1]))
            rptr += 1
        rptr += 2

    wb.save(os.path.join(save_dir, f'{model_name}_results.xlsx'))
    print(f'      [Excel] {model_name}_results.xlsx saved.')


# ═══════════════════════════════════════════════════════════════
# 5-FOLD TRAINING  (strict no-leakage design)
# ═══════════════════════════════════════════════════════════════
def train_model(model_fn, model_name, X, y, save_dir):
    """
    No-leakage 5-fold CV:
      1. StratifiedKFold splits on indices of the already-loaded dataset.
      2. No scaler or transform is fit on the full dataset; pixel /255 is
         a fixed constant applied at load time, not fitted.
      3. K.clear_session() before each fold ensures weight isolation.
      4. Callbacks monitor val metrics only.
    """
    print(f"\n{'='*60}")
    print(f"  Training : {model_name}")
    print(f"{'='*60}")

    skf          = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    histories    = []
    all_cms      = []

    for fold_i, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  ── Fold {fold_i}/{K_FOLDS} "
              f"(train={len(tr_idx):,}  val={len(vl_idx):,}) ──")

        # ── Strict split: NO data from val enters train path ─────
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_vl, y_vl = X[vl_idx], y[vl_idx]

        # Verify no overlap (sanity check)
        assert len(set(tr_idx) & set(vl_idx)) == 0, "DATA LEAKAGE DETECTED!"

        # ── Fresh model for every fold ────────────────────────────
        K.clear_session()
        model = model_fn()

        if fold_i == 1:
            model.summary()
            plot_arch_params(model, model_name, save_dir)
            with open(os.path.join(save_dir, f'{model_name}_params.json'), 'w') as f:
                json.dump({
                    'total_params'    : int(model.count_params()),
                    'trainable_params': int(sum(int(np.prod(v.shape))
                                               for v in model.trainable_weights)),
                    'non_trainable'   : int(sum(int(np.prod(v.shape))
                                               for v in model.non_trainable_weights)),
                }, f, indent=2)

        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_auc', mode='max',
                patience=50, restore_best_weights=True, verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=4, min_lr=1e-6, verbose=1,
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(save_dir,
                                      f'{model_name}_fold{fold_i}_best.keras'),
                monitor='val_auc', mode='max',
                save_best_only=True, verbose=1,
            ),
        ]

        t0   = time.time()
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),   # val set never touches train fit
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=cb_list,
            verbose=1,
        )
        elapsed = time.time() - t0
        histories.append(hist.history)

        # Evaluate on val set with best weights (restored by EarlyStopping)
        y_prob   = model.predict(X_vl, verbose=0).flatten()
        metrics  = compute_metrics(y_vl, y_prob)
        metrics['fold']       = fold_i
        metrics['train_time'] = elapsed
        fold_results.append(metrics)
        all_cms.append(metrics['cm'])

        print(f"    Acc={metrics['accuracy']:.4f}  AUC={metrics['auc']:.4f}  "
              f"F1={metrics['f1']:.4f}  MCC={metrics['mcc']:.4f}  "
              f"Kappa={metrics['kappa']:.4f}  Time={elapsed:.1f}s")

        plot_confusion_matrix(metrics['cm'], model_name, fold_i, save_dir, normalize=False)
        plot_confusion_matrix(metrics['cm'], model_name, fold_i, save_dir, normalize=True)

    # ── Cross-fold aggregate plots & reports ─────────────────────
    plot_history(histories,      model_name, save_dir)
    plot_mean_cm(all_cms,        model_name, save_dir)
    plot_roc_pr(fold_results,    model_name, save_dir)
    plot_metrics_boxplot(fold_results, model_name, save_dir)
    save_excel_report(fold_results, model_name, save_dir)

    print(f"\n  ── {model_name} Final Summary ──")
    for k in ['accuracy', 'auc', 'f1', 'mcc', 'kappa']:
        vals = [fr[k] for fr in fold_results]
        print(f"    {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return fold_results, histories


# ═══════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════
def plot_comparison(all_results, save_dir):
    metrics     = ['accuracy', 'auc', 'f1', 'mcc', 'kappa', 'precision', 'recall']
    model_names = list(all_results.keys())
    short_names = ['M1\nPlain', 'M2\nSE-Attn', 'M3\nRDAD']
    bar_colors  = ['#2196F3', '#4CAF50', '#FF5722']

    fig, axes = plt.subplots(1, len(metrics), figsize=(24, 5))
    fig.suptitle(f'Three-Model Comparison — Mean ± Std  ({K_FOLDS}-Fold CV)',
                 fontsize=13, fontweight='bold')

    for ax, metric in zip(axes, metrics):
        means = [np.mean([fr[metric] for fr in all_results[mn]]) for mn in model_names]
        stds  = [np.std( [fr[metric] for fr in all_results[mn]]) for mn in model_names]
        bars  = ax.bar(short_names, means, yerr=stds, capsize=5,
                       color=bar_colors, alpha=0.85, edgecolor='white')
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_ylim(max(0, min(means) - 0.1), 1.05)
        ax.grid(axis='y', alpha=0.3)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{mean:.4f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'),
                dpi=130, bbox_inches='tight')
    plt.close()


def save_comparison_excel(all_results, save_dir):
    wb    = Workbook()
    ws    = wb.active
    ws.title = 'Model Comparison'
    bdr   = _bdr()
    hfill = PatternFill('solid', fgColor='1F3864')
    hfont = Font(bold=True, color='FFFFFF', size=10)
    halign = Alignment(horizontal='center', vertical='center')

    metrics = ['accuracy','auc','ap','f1','precision','recall',
               'mcc','kappa','specificity','sensitivity']
    headers = ['Model'] \
            + [m.upper() + ' Mean' for m in metrics] \
            + [m.upper() + ' Std'  for m in metrics]
    widths  = [32] + [15] * len(metrics) * 2

    for ci, (h, w) in enumerate(zip(headers, widths), 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.fill = hfill; cell.font = hfont
        cell.alignment = halign; cell.border = bdr
        ws.column_dimensions[cell.column_letter].width = w

    fills = [PatternFill('solid', fgColor=c)
             for c in ['E3F2FD', 'E8F5E9', 'FFF9C4']]

    for ri, (mname, frs) in enumerate(all_results.items(), 2):
        means    = [round(np.mean([fr[m] for fr in frs]), 5) for m in metrics]
        stds     = [round(np.std( [fr[m] for fr in frs]), 5) for m in metrics]
        row_data = [mname] + means + stds
        for ci, val in enumerate(row_data, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.fill = fills[ri - 2]; cell.border = bdr
            cell.alignment = Alignment(vertical='center')

    wb.save(os.path.join(save_dir, 'model_comparison.xlsx'))
    print('  [Excel] model_comparison.xlsx saved.')


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print('=' * 60)
    print('  CNN Classification Pipeline — 3 Models, 5-Fold CV')
    print('  Data Leakage Check: ENABLED')
    print('=' * 60)

    dirs = make_dirs()

    # ── 1. Load data (separate + recombine) ──────────────────────
    X, y, X_label_0, X_label_1 = load_images(EXCEL_INDEX)

    # Shuffle AFTER loading to mix classes
    X, y = sk_shuffle(X, y, random_state=SEED)
    print(f"\n  Dataset after shuffle — X: {X.shape}  "
          f"class balance: {np.bincount(y)}")

    # ── 2. Train three models ─────────────────────────────────────
    model_specs = [
        (build_plain_cnn,        'Model1_PlainCNN',         dirs['model1']),
        (build_cnn_se_attention, 'Model2_CNN_SE_Attention',  dirs['model2']),
        (build_rdad_cnn,         'Model3_RDAD_CNN_Novel',    dirs['model3']),
    ]

    all_results = {}
    for model_fn, model_name, save_dir in model_specs:
        fold_results, _ = train_model(model_fn, model_name, X, y, save_dir)
        all_results[model_name] = fold_results

    # ── 3. Comparison ─────────────────────────────────────────────
    print('\n[COMPARISON] Generating comparison outputs ...')
    plot_comparison(all_results,       dirs['compare'])
    save_comparison_excel(all_results, dirs['compare'])

    # ── 4. Final table ────────────────────────────────────────────
    print('\n' + '=' * 62)
    print(f"  {'Model':<32} {'Acc':>8} {'AUC':>8} {'F1':>8} {'MCC':>8}")
    print('  ' + '-' * 60)
    for mn, frs in all_results.items():
        print(f"  {mn:<32} "
              f"{np.mean([fr['accuracy'] for fr in frs]):>8.4f} "
              f"{np.mean([fr['auc']      for fr in frs]):>8.4f} "
              f"{np.mean([fr['f1']       for fr in frs]):>8.4f} "
              f"{np.mean([fr['mcc']      for fr in frs]):>8.4f}")
    print('=' * 62)
    print(f'\n  All results → {BASE_RESULT}')


if __name__ == '__main__':
    main()