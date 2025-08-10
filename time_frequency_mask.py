import os, random, glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm            # progress bar

from google.colab import drive
drive.mount('/content/drive')

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
SRC_ROOT       = "/content/drive/My Drive/aps360/new_data/training"
DEST_ROOT      = "/content/drive/My Drive/aps360/new_data/augmented"  # <── NEW
OUTPUT_COPIES  = 2          # number of F- and T-masks per original
F_MASK, T_MASK = 20, 40     # max mask sizes
MIN_FRAC       = 0.05       # ≥5 % of rows / cols must be masked
BETA_SHAPE     = 0.4        # MixUp λ ~ Beta(a,a)
LAMBDA_CLIP    = (0.2, 0.8) # keep λ in [0.2, 0.8]

#new only f/t mask

# ===== HELPERS (unchanged) =====
def freq_mask(spec, F=20, min_frac=0.05):
    f = np.random.randint(1, F+1)
    if f / spec.shape[0] < min_frac:
        return freq_mask(spec, F, min_frac)
    f0 = np.random.randint(0, spec.shape[0]-f+1)
    out = spec.copy(); out[f0:f0+f, :] = 0; return out

def time_mask(spec, T=40, min_frac=0.05):
    t = np.random.randint(1, T+1)
    if t / spec.shape[1] < min_frac:
        return time_mask(spec, T, min_frac)
    t0 = np.random.randint(0, spec.shape[1]-t+1)
    out = spec.copy(); out[:, t0:t0+t] = 0; return out

def save(path, arr):
    plt.imsave(path, arr, cmap="magma", vmin=0, vmax=255)

# Defaults if not defined elsewhere
try:
    F_MASK, T_MASK, MIN_FRAC
except NameError:
    F_MASK, T_MASK, MIN_FRAC = 20, 40, 0.05

N_PER_TYPE = 2  # two freq + two time masks per original

# ===== MAIN (2 *_fmask + 2 *_tmask per original) =====
grand_total = 0
for genre in tqdm(os.listdir(SRC_ROOT), desc="Genres"):
    src_dir = os.path.join(SRC_ROOT, genre)
    if not os.path.isdir(src_dir):
        continue

    dest_dir = src_dir  # save next to originals
    exts = (".png", ".jpg", ".jpeg")

    # Treat only originals; ignore previously augmented outputs
    originals = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith(exts) and not any(s in f.lower() for s in ("_fmask", "_tmask"))
    ]

    added = 0
    for fname in originals:
        base, ext = os.path.splitext(fname)
        arr = np.array(Image.open(os.path.join(src_dir, fname)).convert("L"))

        # 2 frequency masks
        for i in range(N_PER_TYPE):
            aug_f = freq_mask(arr, F=F_MASK, min_frac=MIN_FRAC)
            save(os.path.join(dest_dir, f"{base}_fmask{i}{ext}"), aug_f)
            added += 1

        # 2 time masks
        for i in range(N_PER_TYPE):
            aug_t = time_mask(arr, T=T_MASK, min_frac=MIN_FRAC)
            save(os.path.join(dest_dir, f"{base}_tmask{i}{ext}"), aug_t)
            added += 1

    grand_total += added
    print(f"{genre}: {added} augmentations saved in {src_dir}")

print(f"\n✔ Augmentation complete — {grand_total} new images written.")
