"""Train one segmentation model on one PRMI configuration and dump its test predictions.

This is the GPU arm. It exists to answer three questions the synthetic ladder cannot:

  1. **Rank inversion** -- does ranking models by Dice rank them by trait error? (brief section 5,
     right panel)
  2. **The failure-mode fingerprint** -- F12 showed trait error at a fixed Dice depends entirely on
     WHICH failure mode produced that Dice, spanning 0 to 15,500%. So: which failure mode do real
     segmenters actually exhibit? Without this the paper states a range and cannot say where in it
     practice sits.
  3. **The resolution floor** -- can a test set of 7 independent tubes resolve the differences
     between these models at all? `groupeval` answers this; project 02's F29-F31 is the precedent.

DESIGN DECISIONS AND WHY
------------------------
**Empty frames are kept in training.** 48.6% of this configuration's training masks are empty after
resizing, and a segmenter genuinely should output nothing for them. Dropping them would train a
model for a task nobody runs, and would also hide F15's point rather than measure it.

**Dice is reported twice**: over all test frames, and over non-empty frames only. The gap between
them is F15 made concrete on real models -- an empty-empty pair scores 1.0 by convention, so the
all-frames number carries the 0.45 free floor and the non-empty number does not.

**Traits are computed only on non-empty ground truth.** Relative trait error has no denominator
otherwise.

**No DataLoader, cache resident in VRAM as uint8, augmentation on device.** Measured on this machine
in project 02: DataLoader workers were slower (6.5 against 3.3 s/epoch), `channels_last` was 2x
slower, and two concurrent processes gained 1.07x. Those are measurements, not preferences; see
`prmi_dataset.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from prmi_dataset import DEFAULT_CONFIG, DEFAULT_SIZE, load_cache  # noqa: E402

# ImageNet statistics, because every encoder here is ImageNet-pretrained.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# All MIT-licensed and on PyPI, chosen deliberately: project 02's F14 found six of eleven published
# polyp repositories declared no machine-readable licence and none was installable, so none could go
# inside an MIT artifact. `segmentation_models_pytorch` can.
ARCHITECTURES = {
    "unet_r34": ("Unet", "resnet34"),
    "unet_r50": ("Unet", "resnet50"),
    "unetpp_r34": ("UnetPlusPlus", "resnet34"),
    "fpn_r34": ("FPN", "resnet34"),
    "deeplabv3p_r34": ("DeepLabV3Plus", "resnet34"),
    "manet_r34": ("MAnet", "resnet34"),
}


def build_model(arch):
    import segmentation_models_pytorch as smp
    name, encoder = ARCHITECTURES[arch]
    return getattr(smp, name)(encoder_name=encoder, encoder_weights="imagenet",
                              in_channels=3, classes=1)


def normalise(batch_u8):
    x = batch_u8.float().div_(255.0)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x, y, gen):
    """Flips and 90-degree rotations, done on device for the whole batch at once."""
    if torch.rand(1, generator=gen, device=x.device).item() < 0.5:
        x, y = torch.flip(x, [3]), torch.flip(y, [3])
    if torch.rand(1, generator=gen, device=x.device).item() < 0.5:
        x, y = torch.flip(x, [2]), torch.flip(y, [2])
    k = int(torch.randint(0, 4, (1,), generator=gen, device=x.device).item())
    if k:
        x, y = torch.rot90(x, k, [2, 3]), torch.rot90(y, k, [2, 3])
    return x, y


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (num / den).mean()


@torch.no_grad()
def predict(model, imgs, batch=32):
    model.eval()
    out = torch.empty((imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]),
                      dtype=torch.bool, device=imgs.device)
    for i in range(0, imgs.shape[0], batch):
        x = normalise(imgs[i:i + batch])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        out[i:i + batch] = (torch.sigmoid(logits.float()) > 0.5)
    return out


@torch.no_grad()
def dice_per_frame(pred, gt, batch=256):
    """Per-frame Dice on device. Empty-empty scores 1.0, which is the convention F15 is about.

    BATCHED, and it has to be. The first version cast every test frame to float32 at once --
    3,542 x 320 x 320 x 4 bytes = 1.45 GB per operand, plus the product -- while the training cache
    was still resident. It ran out of memory on MAnet at the very end of the unit, AFTER 20 epochs,
    so it threw away 12.5 minutes of completed training. Counting with integer sums on bool slices
    needs no float copy of the whole set.
    """
    out = torch.empty(pred.shape[0], dtype=torch.float64, device=pred.device)
    for i in range(0, pred.shape[0], batch):
        p = pred[i:i + batch].flatten(1)
        g = gt[i:i + batch].flatten(1)
        inter = (p & g).sum(1, dtype=torch.int64)
        tot = p.sum(1, dtype=torch.int64) + g.sum(1, dtype=torch.int64)
        d = torch.where(tot == 0, torch.ones_like(tot, dtype=torch.float64),
                        2.0 * inter.double() / tot.clamp(min=1).double())
        out[i:i + batch] = d
    return out


def train_one(arch, seed, cache_dir, config, size, epochs, batch, lr, device="cuda",
              verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # The TEST cache is loaded only after training, once the training cache has been freed.
    # Holding both during training (4.7 + 2.0 GB on the 736x552 configuration) left UNet++ without
    # room for its decoder: it ran out of memory on its first forward pass.
    tr_x, tr_y, tr_meta = load_cache(cache_dir, config, "train", size, device=device)
    tr_y_f = (tr_y > 127).float()

    model = build_model(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = tr_x.shape[0]
    steps = max(1, n // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        tot = 0.0
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            x = normalise(tr_x[idx])
            y = tr_y_f[idx]
            x, y = augment(x, y, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if verbose and (ep + 1) % 5 == 0:
            print(f"    epoch {ep + 1}/{epochs} loss {tot / steps:.4f} "
                  f"({(time.time() - t0) / (ep + 1):.1f}s/ep)", flush=True)

    # Free the training cache before evaluating. It is 4.1 GB of uint8 plus a float copy of the
    # masks, and nothing after this line needs it.
    del tr_x, tr_y, tr_y_f, opt, sched
    torch.cuda.empty_cache()

    te_x, te_y, te_meta = load_cache(cache_dir, config, "test", size, device=device)
    te_y_b = (te_y > 127)
    pred = predict(model, te_x)
    d = dice_per_frame(pred, te_y_b).cpu().numpy()
    gt_empty = np.array([m["gt_empty_resized"] for m in te_meta])
    pred_empty = (pred.flatten(1).sum(1) == 0).cpu().numpy()

    summary = {
        "arch": arch, "seed": seed, "config": config, "size": size, "epochs": epochs,
        "batch": batch, "lr": lr,
        "n_test": int(len(d)),
        "n_test_nonempty": int((~gt_empty).sum()),
        "mean_dice_all_frames": float(d.mean()),
        "mean_dice_nonempty_only": float(d[~gt_empty].mean()),
        "mean_dice_empty_only": float(d[gt_empty].mean()),
        "pred_empty_rate": float(pred_empty.mean()),
        "train_seconds": float(time.time() - t0),
    }
    return model, pred, d, summary, te_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHITECTURES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--outdir", default="results/prmi_models")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    tag = f"{a.config}__{a.arch}__s{a.seed}"
    print(f"=== {tag} ===", flush=True)
    _model, pred, d, summary, te_meta = train_one(
        a.arch, a.seed, a.cache_dir, a.config, a.size, a.epochs, a.batch, a.lr)

    np.save(os.path.join(a.outdir, tag + "_pred.npy"),
            pred.squeeze(1).cpu().numpy().astype(bool))
    np.save(os.path.join(a.outdir, tag + "_dice.npy"), d)
    with open(os.path.join(a.outdir, tag + "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
