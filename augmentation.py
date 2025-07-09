import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os, random


from google.colab import drive
drive.mount('/content/drive')


def freq_mask(spec, F=20):
    f = np.random.randint(0, F)
    f0 = np.random.randint(0, spec.shape[0] - f)
    spec[f0:f0+f, :] = 0
    return spec

def time_mask(spec, T=40):
    t = np.random.randint(0, T)
    t0 = np.random.randint(0, spec.shape[1] - t)
    spec[:, t0:t0+t] = 0
    return spec


###############change this##################
root = "/content/drive/My Drive/aps360/data_folder/training"  
output_copies = 2
F_mask = 20
T_mask = 40


for genre in os.listdir(root):
    genre_dir = os.path.join(root, genre)
    if not os.path.isdir(genre_dir):
        continue
    
    pngs = [f for f in os.listdir(genre_dir) if f.endswith(".png")]


    for fname in os.listdir(genre_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(genre_dir, fname)
        img = Image.open(path).convert("L")
        arr = np.array(img)

        base, ext = os.path.splitext(fname)
        mask_fns = [freq_mask, time_mask]

        for i, mask_fn in enumerate(mask_fns, start=1):
            aug = arr.copy()
            # mask
            if mask_fn is freq_mask:
                aug = freq_mask(aug, F_mask)
            else:
                aug = time_mask(aug, T_mask)

            out_name = f"{base}_aug{i+1}{ext}"
            out_path = os.path.join(genre_dir, out_name)

            plt.imsave(out_path, aug, cmap='magma', origin='upper', vmin=0, vmax=255)  # writes an RGB PNG


            partner = random.choice([p for p in pngs if p != fname])
            arr2    = np.array(Image.open(os.path.join(genre_dir, partner)).convert("L"))
            lam     = np.random.beta(0.4, 0.4)

            mixed = (lam * arr + (1 - lam) * arr2).astype(np.uint8)
            mix_name = f"{base}_mix_{os.path.splitext(partner)[0]}_{lam:.2f}.png"
            mix_path = os.path.join(genre_dir, mix_name)
            plt.imsave(mix_path, mixed,
                      cmap='magma', origin='upper',
                      vmin=0, vmax=255)


    print(f"Augmented {genre}: +{output_copies} images each.")

print("All done!")
