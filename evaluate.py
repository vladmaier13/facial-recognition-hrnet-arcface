import argparse
import random
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, auc, confusion_matrix,
                             roc_curve)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from dataset import build_dataset
from losses import ArcLoss
from network import ArcLayer, L2Normalization

# ========================= Helper utilities ============================

def _to_uint8(img, eps=1e-8):
    """
    Convertește orice tensor imagine în uint8 afișabil.
    – Dacă e deja uint8 → return.
    – Altfel rescalează (min-max) la [0,255].
    """
    if img.dtype == np.uint8:
        return img

    img = np.asarray(img, dtype=np.float32)
    v_min, v_max = img.min(), img.max()
    if v_max - v_min < eps:      # imagine constantă
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - v_min) / (v_max - v_min)   # [0,1]
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def _save_debug(img, fname_np="debug.npy", fname_hist="debug_hist.png"):
    """Salvează tensor brut + histogramă valorilor."""
    np.save(fname_np, img)
    plt.figure()
    flat = img.flatten()
    plt.hist(flat, bins=50, color="gray")
    plt.title("Histogramă valori brute")
    plt.xlabel("Valoare pixel"); plt.ylabel("Frecvență")
    plt.tight_layout(); plt.savefig(fname_hist); plt.close()
    print(f"🔍 Tensor brut salvat în '{fname_np}' + histogramă în '{fname_hist}'.")


def _plot_confusion_examples(imgs, y_true, y_pred,
                             max_examples=3, prefix="confusion_example"):
    """Exemple de clasificări greșite (confuzii)."""
    mis = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    if not mis:
        print("✅ Nicio confuzie.")
        return

    shown = 0
    for idx in mis:
        if shown >= max_examples:
            break
        true_lbl, pred_lbl = y_true[idx], y_pred[idx]
        try:
            alt_idx = next(j for j, lbl in enumerate(y_true) if lbl == pred_lbl and j != idx)
        except StopIteration:
            alt_idx = idx

        # SALVĂM tensorul brut al primului exemplu confuz pentru debug
        if shown == 0:
            _save_debug(imgs[idx])

        img_a = _to_uint8(imgs[idx])
        img_b = _to_uint8(imgs[alt_idx])

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(img_a); axes[0].set_title(f"Corect: {true_lbl}"); axes[0].axis("off")
        axes[1].imshow(img_b); axes[1].set_title(f"Prezicere: {pred_lbl}"); axes[1].axis("off")
        plt.suptitle(f"Confuzie #{shown + 1}")
        plt.tight_layout()
        fname = f"{prefix}_{shown + 1}.png"
        plt.savefig(fname); plt.show()
        print(f"🖼️ Confuzie salvată în '{fname}'.")
        shown += 1


def _plot_correct_examples(imgs, y_true, y_pred,
                           max_examples=3, prefix="correct_example"):
    """Exemple corecte: două imagini diferite din aceeași clasă."""
    correct = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
    if len(correct) < 2:
        print("ℹ️ Prea puține corecte pentru vizualizare.")
        return

    shown, used = 0, set()
    for idx in correct:
        if shown >= max_examples or idx in used:
            continue
        label = y_true[idx]
        try:
            alt_idx = next(j for j in correct
                           if j != idx and j not in used and y_true[j] == label)
        except StopIteration:
            continue

        # SALVĂM tensor brut al primului exemplu corect (dacă încă nu l-am salvat)
        if not os.path.exists("debug.npy"):   # dacă nu s-a salvat la confuzie
            _save_debug(imgs[idx])

        img_a = _to_uint8(imgs[idx])
        img_b = _to_uint8(imgs[alt_idx])

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(img_a); axes[0].set_title(f"Corect: {label}"); axes[0].axis("off")
        axes[1].imshow(img_b); axes[1].set_title(f"Alt exemplu: {label}"); axes[1].axis("off")
        plt.suptitle(f"Predicție corectă #{shown + 1}")
        plt.tight_layout()
        fname = f"{prefix}_{shown + 1}.png"
        plt.savefig(fname); plt.show()
        print(f"✅ Exemplu corect salvat în '{fname}'.")
        used.update([idx, alt_idx])
        shown += 1


def _plot_top_confusions_heatmap(cm, top_n=20, fname="heatmap_top_confusions.png"):
    if cm.size == 0:
        return
    errors = cm.copy().astype(float); np.fill_diagonal(errors, 0)
    tot = errors.sum(1) + errors.sum(0)
    if (tot == 0).all(): return
    idx = np.argsort(tot)[::-1][:min(top_n, (tot > 0).sum())]
    sub = cm[np.ix_(idx, idx)].astype(float)
    sub = sub / sub.sum(1, keepdims=True)
    plt.figure(figsize=(1+0.4*len(idx), 0.8+0.4*len(idx)))
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=idx, yticklabels=idx)
    plt.title(f"Top {len(idx)} clase confundate (normalizat pe rând)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(fname); plt.show()
    print(f"🔥 Heatmap în '{fname}'.")


# ========================= Model loading ===============================

def load_model_with_custom_objects(path):
    print(f"🔄 Încărcăm modelul din: {path}")
    return tf.keras.models.load_model(
        path,
        custom_objects={"ArcLoss": ArcLoss,
                        "ArcLayer": ArcLayer,
                        "L2Normalization": L2Normalization}
    )

# ========================= Evaluations =================================

def evaluate_classification(model, ds, use_softmax):
    if use_softmax:
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
    else:
        model.compile(loss=ArcLoss(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
    loss, acc = model.evaluate(ds); print(f"\n🎯 Acc={acc:.4f} | Loss={loss:.4f}")

    y_true, y_pred, imgs = [], [], []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0)
        y_true.extend(np.argmax(yb.numpy(), 1))
        y_pred.extend(np.argmax(p, 1))
        imgs.extend(xb.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.title("Matrice de confuzie – toate clasele")
    plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.show()

    _plot_top_confusions_heatmap(cm)
    _plot_confusion_examples(imgs, y_true, y_pred)
    _plot_correct_examples(imgs, y_true, y_pred)

# ---------- ArcFace similarity evaluation ------------------------------

def generate_pairs(emb, lbl, n=1000):
    lbl2idx = {}
    for i, l in enumerate(lbl): lbl2idx.setdefault(l, []).append(i)
    pairs, y = [], []
    for _ in range(n//2):                # pozitive
        l = random.choice([l for l in lbl2idx if len(lbl2idx[l])>1])
        i1, i2 = random.sample(lbl2idx[l], 2)
        pairs.append((emb[i1], emb[i2])); y.append(1)
    L = list(lbl2idx.keys())
    for _ in range(n//2):                # negative
        l1, l2 = random.sample(L, 2)
        i1 = random.choice(lbl2idx[l1]); i2 = random.choice(lbl2idx[l2])
        pairs.append((emb[i1], emb[i2])); y.append(0)
    return pairs, y

def evaluate_arcface_similarity(model, ds):
    emb, lbl = [], []
    for xb, yb in tqdm(ds, desc="🔄 embeddings"):
        emb.append(model(xb).numpy())
        lbl.append(np.argmax(yb.numpy(), 1))
    emb, lbl = np.concatenate(emb), np.concatenate(lbl)
    pairs, y = generate_pairs(emb, lbl)
    sims = [cosine_similarity([a], [b])[0,0] for a,b in pairs]
    fpr, tpr, _ = roc_curve(y, sims); auc_val = auc(fpr, tpr)
    print(f"\n📈 ROC AUC={auc_val:.4f}")
    plt.figure(); plt.plot(fpr, tpr, lw=2, label=f"AUC={auc_val:.4f}")
    plt.plot([0,1],[0,1], lw=2, ls="--"); plt.legend(); plt.grid()
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout()
    plt.savefig("roc_curve.png"); plt.show()
    # histogram
    plt.figure()
    plt.hist([s for s,l in zip(sims,y) if l==1], bins=30, alpha=.6, label="Pozitive")
    plt.hist([s for s,l in zip(sims,y) if l==0], bins=30, alpha=.6, label="Negative")
    plt.legend(); plt.xlabel("Cosine"); plt.ylabel("Freq"); plt.tight_layout()
    plt.savefig("similaritate_cosine.png"); plt.show()

# ========================= Main ========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--test_record",
                    default="D:\\Licenta\\arcface-main\\faces_emore\\split_dataset\\test.record")
    ap.add_argument("--num_ids", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--softmax", action="store_true")
    args = ap.parse_args()

    ds = build_dataset(args.test_record, batch_size=args.batch_size,
                       one_hot_depth=args.num_ids, training=False, buffer_size=4096)
    model = load_model_with_custom_objects(args.model_path)
    evaluate_classification(model, ds, args.softmax)
    if not args.softmax:
        evaluate_arcface_similarity(model, ds)

if __name__ == "__main__":
    main()
