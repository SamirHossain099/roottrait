"""Build a resident uint8 tensor cache of one PRMI imaging configuration.

WHY A CACHE, AND WHY ON THE GPU
-------------------------------
Project 02 measured the alternatives on this exact machine and the results were counter-intuitive
enough to be worth restating rather than rediscovering:

  * a `DataLoader` with workers was SLOWER than none (6.5 against 3.3 s/epoch) -- Windows spawn plus
    host-to-device copies cost more than a trivial `__getitem__` over an in-RAM cache;
  * `channels_last` was 2x SLOWER at this batch size and resolution;
  * two concurrent training processes gained only 1.07x, so the workload is near compute-bound.

So the whole configuration is decoded once, resized once, and held as uint8 in VRAM. Training then
indexes it directly and does augmentation and normalisation on device. uint8 is what makes this fit:
10,087 frames at 320x320x3 is 3.1 GB as uint8 and 12.4 GB as float32.

WHY ONE CONFIGURATION AT A TIME
-------------------------------
F16. PRMI's species folders encode the imaging setup, peanut and sesame each appear at two, and
every trait in this project is measured in PIXELS while DPI varies 120/150/300. Pooling
configurations would compare measurements that are not on a common scale, which is F10's thickness
confound wearing different clothes. Peanut is the configuration of choice precisely because it
appears twice, so the second configuration is a replication rather than a different experiment.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from prmi_ingest import index_directory  # noqa: E402
from prmi_load import open_source, pair_images_and_masks, to_boolean_mask  # noqa: E402

DEFAULT_CONFIG = "peanut_640x480_DPI120"
DEFAULT_SIZE = 320


def _resize_u8(a, size):
    """Nearest-neighbour resize via torch, so images and masks use the same geometry.

    Masks must be nearest -- bilinear would create intermediate values and the whole point of
    `to_boolean_mask` is that a mask is binary. Images use the same path so that a pixel in the
    image and the corresponding pixel in the mask came from the same source location.
    """
    t = torch.from_numpy(np.ascontiguousarray(a))
    if t.ndim == 2:
        t = t[None, None]
    else:
        t = t.permute(2, 0, 1)[None]
    out = torch.nn.functional.interpolate(t.float(), size=(size, size), mode="nearest")
    return out[0].to(torch.uint8)


def build_cache(root, config, split, size=DEFAULT_SIZE, limit=None, verbose=True,
                report_every=2000):
    """Decode one (config, split) into uint8 tensors plus per-frame metadata."""
    records, _ = index_directory(root)
    pairs, _, _, _ = pair_images_and_masks(records)
    sel = [p for p in pairs
           if p["mask"]["config"] == config and p["mask"]["official_split"] == split]
    sel.sort(key=lambda p: p["mask"]["path"])          # deterministic order, independent of the OS
    if limit:
        sel = sel[:limit]
    if not sel:
        raise ValueError(f"no pairs for config={config!r} split={split!r}")

    imgs = torch.empty((len(sel), 3, size, size), dtype=torch.uint8)
    msks = torch.empty((len(sel), 1, size, size), dtype=torch.uint8)
    meta = []
    t0 = time.time()
    with open_source(root) as src:
        for i, p in enumerate(sel):
            raw_i = src.read(p["image"]["path"])
            a = _decode_rgb(raw_i)
            m = to_boolean_mask(src.read(p["mask"]["path"]), path=p["mask"]["path"])
            imgs[i] = _resize_u8(a, size)
            msks[i] = _resize_u8(m.astype(np.uint8) * 255, size)
            meta.append({
                "path": p["mask"]["path"],
                "group": p["mask"]["group"],
                "session": p["mask"]["session"],
                "gt_fg_px_native": int(m.sum()),
                "gt_empty": bool(m.sum() == 0),
                "native_h": int(m.shape[0]), "native_w": int(m.shape[1]),
            })
            if verbose and (i + 1) % report_every == 0:
                r = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"  {i + 1}/{len(sel)}  ({r:.0f}/s)", flush=True)

    # Resizing can empty a mask that was non-empty natively (a one-pixel root at 480 -> 320). Record
    # emptiness AFTER resizing too: that is what the model actually sees and is scored against, and
    # silently mixing the two definitions would make the empty rate unreproducible.
    for i, mm in enumerate(meta):
        mm["gt_fg_px_resized"] = int((msks[i] > 127).sum())
        mm["gt_empty_resized"] = bool(mm["gt_fg_px_resized"] == 0)
    return imgs, msks, meta


def _decode_rgb(raw):
    a = _decode_image_any(raw)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return a[..., :3]


def _decode_image_any(raw):
    """Decode without the binary-mask checks `_decode` applies to the single channel it keeps."""
    import io as _io

    from imageio.v3 import imread
    return np.asarray(imread(_io.BytesIO(raw)))


def cache_path(outdir, config, split, size):
    return os.path.join(outdir, f"{config}_{split}_{size}.pt")


def save_cache(outdir, config, split, size, imgs, msks, meta):
    os.makedirs(outdir, exist_ok=True)
    p = cache_path(outdir, config, split, size)
    torch.save({"images": imgs, "masks": msks, "config": config, "split": split, "size": size},
               p)
    with open(p.replace(".pt", "_meta.json"), "w") as fh:
        json.dump(meta, fh)
    return p


def load_cache(outdir, config, split, size, device="cuda"):
    p = cache_path(outdir, config, split, size)
    d = torch.load(p, map_location="cpu", weights_only=False)
    with open(p.replace(".pt", "_meta.json")) as fh:
        meta = json.load(fh)
    return d["images"].to(device), d["masks"].to(device), meta


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--outdir", default="data/cache")
    a = ap.parse_args()

    for split in a.splits:
        print(f"=== {a.config} / {split} ===")
        imgs, msks, meta = build_cache(a.dir, a.config, split, size=a.size)
        p = save_cache(a.outdir, a.config, split, a.size, imgs, msks, meta)
        gb = (imgs.numel() + msks.numel()) / 1e9
        n_empty = sum(m["gt_empty_resized"] for m in meta)
        print(f"  {len(meta)} frames, {gb:.2f} GB uint8, "
              f"{n_empty} empty after resize ({n_empty / len(meta):.1%}), "
              f"{len({m['group'] for m in meta})} tubes -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
