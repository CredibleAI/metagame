"""Meta-ConceptAttention single-token segmentation evaluation (Table 3 + Fig 13).

Evaluates ConceptAttention vs Meta-ConceptAttention on FLUX.1 [schnell] across
COCO, Pascal VOC, and ImageNet-Segmentation. Two methods, two tracks per
dataset.

Methods.
  CA       — `pipeline.encode_image(softmax=True)` → softmax across the full
             coalition (paper baseline, Helbling et al. 2025).
  Meta-CA  — single `softmax=False` forward pass cached as raw scores; the
             Shapley diagonal sv[t, t] is computed by streaming over coalitions
             of every size 1..N, batching coalitions per size into chunks and
             applying the per-chunk softmax on GPU. Constant memory wrt 2^N.
             Math: sv[t, t] = Σ_{T ∋ t} (|T|-1)!(N-|T|)!/N! · softmax_{T∪ctx}(raw)[t]

Datasets.
  Pascal VOC 2012 val      — 20 single-T5-token thing classes; full 20-class
                             set lands in every image's player set.
  MS COCO 2017 panoptic val   — 53 single-T5-token thing classes (bi-token classes
                             excluded). Eligibility: ≥ 1 single-token GT thing.
  ImageNet-Seg (IJCV 2014) — Guillaumin's gtsegs_ijcv.mat, 4276 images filtered
                             to single-token simplified names (~3535 images).
                             Binary fg/bg masks; one fg class per image.

Tracks.
  single — paper Table 3 binary threshold-at-mean over (image, in-vocab class).
  multi  — paper Table 3 argmax over [bg + thing channels], scored on pixels
           whose ground-truth lies in the in-vocabulary thing set (paper §D.3).

Player layout.
  --total-players N    — pad each image's player set to exactly N single-token
                         thing classes (GT-present + random distractors,
                         deterministic by image_id). N=20 = canonical (Pascal VOC's
                         full 20-class set; MS COCO's GT + distractors). N=0
                         disables padding (image-conditional, GT-present only).
  --n-distractors d    — variable-size: per-image total = n_present + d.

Ablations (paper App. Fig 13).
  --prompt             — feed FLUX an artificial prompt 'a {cls1}, a {cls2}, …'
                         built from GT-present classes (default: empty prompt).
  --decoupled          — concept_self_attention=False for both forward passes
                         (= without cross-concept attention).
  --layer-indices …    — override DEFAULT_LAYERS=range(9, 19); paper uses
                         range(14, 19) for the "last 5 layers only" ablation.

Hyper-parameters (locked).
  layer_indices    = range(9, 19)            # paper text "last 10 of 18 MMATTN"
  noise_timestep   = 2                       # paper §C.4
  num_steps        = 4                       # flux-schnell default
  width = height   = 1024                    # flux-schnell default
  bg_concepts      = ["background", "floor", "grass", "tree", "sky"]

Outputs.
  results/<variant>/
    shard_<i>_of_<n>.json   — per-image rows (one per shard)
    metrics.json            — pooled per-track metrics + per-class breakdowns

Usage.
  conda run -n metaconceptattention python main.py --dataset mscoco
  conda run -n metaconceptattention python main.py \\
      --dataset mscoco --shard-i 0 --shard-n 10
  conda run -n metaconceptattention python main.py --dataset mscoco --aggregate
"""
import argparse
import dataclasses
import itertools
import json
import math
import os
import random
import socket
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import Counter
from glob import glob

import numpy as np
import PIL.Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm

try:
    from concept_attention import ConceptAttentionFluxPipeline
except ImportError:
    ConceptAttentionFluxPipeline = None


HERE = os.path.dirname(os.path.abspath(__file__))
HF_CACHE_ROOT = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
MSCOCO_ROOT = os.path.join(HF_CACHE_ROOT, "mscoco")
PASCALVOC_ROOT  = os.path.join(HF_CACHE_ROOT, "pascalvoc")
RESULTS_BASE = os.path.join(HERE, "results")

IMAGENETSEG_DIR     = os.path.join(HF_CACHE_ROOT, "imagenetseg")
IMAGENETSEG_MAT     = os.path.join(IMAGENETSEG_DIR, "gtsegs_ijcv.mat")
IMAGENETSEG_CLASSES = os.path.join(IMAGENETSEG_DIR, "imagenet_class_map.json")
IMAGENETSEG_IMG_DIR  = os.path.join(IMAGENETSEG_DIR, "images")
IMAGENETSEG_MASK_DIR = os.path.join(IMAGENETSEG_DIR, "masks")

MSCOCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
MSCOCO_PANOPTIC_ANN_URL = ("http://images.cocodataset.org/annotations/"
                         "panoptic_annotations_trainval2017.zip")
PASCALVOC_URL = ("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
               "VOCtrainval_11-May-2012.tar")

DEFAULT_LAYERS = list(range(9, 19))
BG_CONCEPTS = ["background", "floor", "grass", "tree", "sky"]
EVAL_RES = (224, 224)

# Pascal VOC class names with the paper's T5-friendly remapping (all single-T5-token).
# Position 0 = background; we shift by +1 internally so 0 means "unlabeled".
PASCALVOC_CLASSES_NAMES = [
    "background", "plane", "bike", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "table", "dog", "horse", "motorcycle", "person",
    "pot", "sheep", "sofa", "train", "monitor",
]

METHOD_KEYS = ("ca", "meta_ca")
METHOD_LABELS = {"ca": "ConceptAttention", "meta_ca": "Meta-ConceptAttention"}


class SkipImage(Exception):
    """Signals an image should be skipped (e.g. n_present_players > total_players)."""


def set_seeds(seed_value):
    import torch
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Datasets
# =============================================================================

def _wget(url, dst_path):
    if os.path.exists(dst_path):
        return
    print(f"  downloading  {url}\n           →   {dst_path}", flush=True)
    subprocess.check_call(["wget", "--progress=bar:force:noscroll", url,
                           "-O", dst_path])


def _unzip(zip_path, dst_dir):
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dst_dir)


def _untar(tar_path, dst_dir):
    with tarfile.open(tar_path) as t:
        t.extractall(dst_dir)


def ensure_mscoco_downloads():
    os.makedirs(MSCOCO_ROOT, exist_ok=True)
    val_dir = os.path.join(MSCOCO_ROOT, "val2017")
    ann_dir = os.path.join(MSCOCO_ROOT, "annotations")
    pan_json = os.path.join(ann_dir, "panoptic_val2017.json")
    pan_mask_dir = os.path.join(ann_dir, "panoptic_val2017")
    if not os.path.isdir(val_dir) or len(os.listdir(val_dir)) < 5000:
        zip_path = os.path.join(MSCOCO_ROOT, "val2017.zip")
        _wget(MSCOCO_VAL2017_URL, zip_path)
        _unzip(zip_path, MSCOCO_ROOT)
    if not os.path.isfile(pan_json) or not os.path.isdir(pan_mask_dir):
        zip_path = os.path.join(MSCOCO_ROOT, "panoptic_annotations_trainval2017.zip")
        _wget(MSCOCO_PANOPTIC_ANN_URL, zip_path)
        _unzip(zip_path, MSCOCO_ROOT)
        nested_zip = os.path.join(ann_dir, "panoptic_val2017.zip")
        if os.path.exists(nested_zip):
            _unzip(nested_zip, ann_dir)
    return val_dir, pan_json, pan_mask_dir


def ensure_pascalvoc_downloads():
    os.makedirs(PASCALVOC_ROOT, exist_ok=True)
    base = os.path.join(PASCALVOC_ROOT, "VOCdevkit", "VOC2012")
    jpg_dir = os.path.join(base, "JPEGImages")
    mask_dir = os.path.join(base, "SegmentationClass")
    val_split = os.path.join(base, "ImageSets", "Segmentation", "val.txt")
    if not os.path.isfile(val_split) or not os.path.isdir(mask_dir):
        tar_path = os.path.join(PASCALVOC_ROOT, "VOCtrainval_11-May-2012.tar")
        _wget(PASCALVOC_URL, tar_path)
        _untar(tar_path, PASCALVOC_ROOT)
    return jpg_dir, mask_dir, val_split


def _strip_panoptic_suffix(name):
    s = name
    for suf in ("-merged", "-stuff", "-other"):
        s = s.replace(suf, "")
    return s.strip("-")


def _t5_tokens(name, tokenizer):
    return tokenizer.encode(name, add_special_tokens=False)


def _t5_token_strings(name, tokenizer):
    ids = _t5_tokens(name, tokenizer)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    return [p.lstrip("▁") or p for p in pieces]


def _decode_panoptic_to_semantic(pan_png_path, segments_info):
    pan_rgb = np.asarray(PIL.Image.open(pan_png_path).convert("RGB"))
    seg_id = (pan_rgb[..., 0].astype(np.int64)
              + pan_rgb[..., 1].astype(np.int64) * 256
              + pan_rgb[..., 2].astype(np.int64) * 256 ** 2)
    semantic = np.zeros_like(seg_id, dtype=np.int32)
    for s in segments_info:
        semantic[seg_id == int(s["id"])] = int(s["category_id"])
    return semantic


def _decode_pascalvoc_target(png_path):
    """Pascal VOC SegmentationClass PNG → cat_id mask. 0=unlabeled (was 255),
    1=background (was 0), 2..21=thing classes (was 1..20)."""
    arr = np.asarray(PIL.Image.open(png_path), dtype=np.int64)
    out = np.zeros_like(arr, dtype=np.int32)
    keep = (arr >= 0) & (arr <= 20)
    out[keep] = arr[keep].astype(np.int32) + 1
    return out


@dataclasses.dataclass
class DatasetCtx:
    name: str
    eligible: list
    decode_target: callable
    class_id_to_short: dict
    class_id_to_tokens: dict
    bg_concepts: list
    bg_cat_ids: list
    things_kept: list
    stuff_kept: list


def load_mscoco(args, tokenizer):
    print(f"MS COCO root: {MSCOCO_ROOT}", flush=True)
    val_dir, pan_json_path, pan_mask_dir = ensure_mscoco_downloads()
    with open(pan_json_path) as f:
        data = json.load(f)
    cats = {int(c["id"]): c for c in data["categories"]}
    anns = {int(a["image_id"]): a for a in data["annotations"]}

    things, stuff = [], []
    for cid, cat in sorted(cats.items()):
        raw = cat["name"]
        short = _strip_panoptic_suffix(raw)
        n_tok = len(_t5_tokens(short, tokenizer))
        if int(cat.get("isthing", 1)) == 1:
            if n_tok == 1:
                things.append((cid, raw, short, 1))
        elif n_tok == 1:
            stuff.append((cid, raw, short, 1))
    print(f"  things kept (single-token): {len(things)}/80", flush=True)

    stuff_name_to_cid = {s[2]: s[0] for s in stuff}
    bg_concepts = list(BG_CONCEPTS)
    bg_cat_ids = [stuff_name_to_cid.get(c, 0) for c in bg_concepts]
    print(f"  bg concepts: {list(zip(bg_concepts, bg_cat_ids))}", flush=True)

    thing_ids_set = {t[0] for t in things}
    class_id_to_short = {t[0]: t[2] for t in things}
    class_id_to_short.update({s[0]: s[2] for s in stuff})
    class_id_to_tokens = {t[0]: tuple(_t5_token_strings(t[2], tokenizer))
                          for t in things}
    class_id_to_tokens.update({s[0]: tuple(_t5_token_strings(s[2], tokenizer))
                               for s in stuff})

    eligible = []
    for image_id, ann in anns.items():
        present = []
        for s in ann["segments_info"]:
            cid = int(s["category_id"])
            if int(cats[cid].get("isthing", 1)) == 1 and cid in thing_ids_set:
                present.append(cid)
        present = sorted(set(present))
        if not present:
            continue
        png = ann["file_name"]
        eligible.append({
            "image_id": int(image_id),
            "jpg_path": os.path.join(val_dir, png.rsplit(".", 1)[0] + ".jpg"),
            "mask_path": os.path.join(pan_mask_dir, png),
            "present_class_ids": present,
            "meta": {"segments_info": ann["segments_info"]},
        })

    def _decode(entry):
        return _decode_panoptic_to_semantic(entry["mask_path"],
                                            entry["meta"]["segments_info"])
    return DatasetCtx(name="mscoco", eligible=eligible, decode_target=_decode,
                      class_id_to_short=class_id_to_short,
                      class_id_to_tokens=class_id_to_tokens,
                      bg_concepts=bg_concepts, bg_cat_ids=bg_cat_ids,
                      things_kept=things, stuff_kept=stuff)


def load_pascalvoc(args, tokenizer):
    print(f"Pascal VOC root: {PASCALVOC_ROOT}", flush=True)
    jpg_dir, mask_dir, val_split = ensure_pascalvoc_downloads()
    things = []
    for i, name in enumerate(PASCALVOC_CLASSES_NAMES[1:], start=2):
        n_tok = len(_t5_tokens(name, tokenizer))
        if n_tok == 1:
            things.append((i, name, name, 1))
    stuff = [(1, "background", "background", 1)]

    bg_concepts = list(BG_CONCEPTS)
    bg_cat_ids = [1] * len(bg_concepts)

    class_id_to_short = {t[0]: t[2] for t in things}
    class_id_to_tokens = {t[0]: tuple(_t5_token_strings(t[2], tokenizer))
                          for t in things}

    with open(val_split) as f:
        val_ids = [line.strip() for line in f if line.strip()]
    print(f"  Pascal VOC val images: {len(val_ids)}", flush=True)

    thing_ids_set = {t[0] for t in things}
    eligible = []
    for image_id_str in val_ids:
        png = os.path.join(mask_dir, f"{image_id_str}.png")
        jpg = os.path.join(jpg_dir, f"{image_id_str}.jpg")
        if not os.path.isfile(png) or not os.path.isfile(jpg):
            continue
        target = _decode_pascalvoc_target(png)
        present = sorted({int(c) for c in np.unique(target)
                          if int(c) >= 2 and int(c) in thing_ids_set})
        if not present:
            continue
        eligible.append({
            "image_id": image_id_str,
            "jpg_path": jpg, "mask_path": png,
            "present_class_ids": present, "meta": {},
        })

    def _decode(entry):
        return _decode_pascalvoc_target(entry["mask_path"])
    return DatasetCtx(name="pascalvoc", eligible=eligible, decode_target=_decode,
                      class_id_to_short=class_id_to_short,
                      class_id_to_tokens=class_id_to_tokens,
                      bg_concepts=bg_concepts, bg_cat_ids=bg_cat_ids,
                      things_kept=things, stuff_kept=stuff)


def _ensure_imagenetseg_cache():
    import h5py
    if not os.path.isfile(IMAGENETSEG_MAT):
        sys.exit(f"missing {IMAGENETSEG_MAT}; download via:\n"
                 f"  wget http://calvin-vision.net/bigstuff/proj-imagenet/data/"
                 f"gtsegs_ijcv.mat -O {IMAGENETSEG_MAT}")
    if not os.path.isfile(IMAGENETSEG_CLASSES):
        sys.exit(f"missing {IMAGENETSEG_CLASSES}")
    os.makedirs(IMAGENETSEG_IMG_DIR, exist_ok=True)
    os.makedirs(IMAGENETSEG_MASK_DIR, exist_ok=True)
    with open(IMAGENETSEG_CLASSES) as cf:
        cls_map = json.load(cf)["categories"]
    from nltk.corpus import wordnet as wn
    f = h5py.File(IMAGENETSEG_MAT, "r")
    n = f['/value/id'].shape[0]
    items = []
    n_extracted = 0
    for i in range(n):
        img_path = os.path.join(IMAGENETSEG_IMG_DIR, f"{i}.png")
        msk_path = os.path.join(IMAGENETSEG_MASK_DIR, f"{i}.png")
        if not (os.path.isfile(img_path) and os.path.isfile(msk_path)):
            img = np.array(f[f['/value/img'][i, 0]]).transpose((2, 1, 0))
            target = np.array(f[f[f['/value/gt'][i, 0]][0, 0]]).transpose((1, 0))
            PIL.Image.fromarray(img).save(img_path)
            PIL.Image.fromarray(target.astype(np.uint8)).save(msk_path)
            n_extracted += 1
        id_bytes = f[f['/value/id'][i, 0]]
        synset_code = b"".join(id_bytes).decode('utf-16').strip()
        try:
            offset = int(synset_code[1:].split('_')[0])
            syn = wn.synset_from_pos_and_offset(synset_code[0], offset)
            raw_name = syn.lemmas()[0].name().replace('_', ' ')
        except Exception:
            raw_name = None
        simplified = cls_map.get(raw_name) if raw_name else None
        items.append({"image_id": i, "img_path": img_path,
                      "mask_path": msk_path,
                      "raw_name": raw_name, "simplified_name": simplified})
    f.close()
    if n_extracted:
        print(f"  extracted {n_extracted} new image/mask pairs to "
              f"{IMAGENETSEG_IMG_DIR}", flush=True)
    return items


def _decode_imagenetseg_mask(mask_path, fg_cid):
    arr = np.array(PIL.Image.open(mask_path))
    out = np.ones_like(arr, dtype=np.int32)
    out[arr > 0] = int(fg_cid)
    return out


def load_imagenetseg(args, tokenizer):
    print(f"ImageNet-Seg root: {IMAGENETSEG_DIR}", flush=True)
    items = _ensure_imagenetseg_cache()
    print(f"  cache items: {len(items)}", flush=True)

    name_to_cid = {}
    things = []
    n_no_lookup = 0
    for it in items:
        name = it["simplified_name"]
        if name is None:
            n_no_lookup += 1
            continue
        if name in name_to_cid:
            continue
        n_tok = len(_t5_tokens(name, tokenizer))
        if n_tok != 1:
            continue
        cid = len(things) + 2
        name_to_cid[name] = cid
        things.append((cid, name, name, 1))
    print(f"  unique single-token classes: {len(things)} "
          f"(no-lookup: {n_no_lookup})", flush=True)

    bg_concepts = list(BG_CONCEPTS)
    bg_cat_ids = [1] * len(bg_concepts)
    stuff = [(1, "background", "background", 1)]
    class_id_to_short = {t[0]: t[2] for t in things}
    class_id_to_short.update({s[0]: s[2] for s in stuff})
    class_id_to_tokens = {t[0]: tuple(_t5_token_strings(t[2], tokenizer))
                          for t in things}

    eligible = []
    for it in items:
        cid = name_to_cid.get(it["simplified_name"])
        if cid is None:
            continue
        eligible.append({
            "image_id": it["image_id"],
            "jpg_path": it["img_path"],
            "mask_path": it["mask_path"],
            "present_class_ids": [cid],
            "meta": {"raw_name": it["raw_name"], "fg_cid": cid},
        })
    print(f"  eligible: {len(eligible)}", flush=True)

    def _decode(entry):
        return _decode_imagenetseg_mask(entry["mask_path"], entry["meta"]["fg_cid"])

    return DatasetCtx(name="imagenetseg", eligible=eligible, decode_target=_decode,
                      class_id_to_short=class_id_to_short,
                      class_id_to_tokens=class_id_to_tokens,
                      bg_concepts=bg_concepts, bg_cat_ids=bg_cat_ids,
                      things_kept=things, stuff_kept=stuff)


# =============================================================================
# Player layout + forward passes
# =============================================================================

def build_player_layout(present_class_ids, ctx, total_players, image_id):
    """[bg_0..bg_{n_bg-1}] + [present-class single tokens]
                          + [random distractor single tokens to reach total_players]."""
    concept_list = list(ctx.bg_concepts)
    n_bg = len(concept_list)
    context_idx = tuple(range(n_bg))
    class_to_players = {}
    class_order = []
    for cid in present_class_ids:
        idxs = []
        for tok in ctx.class_id_to_tokens[cid]:
            idxs.append(len(concept_list))
            concept_list.append(tok)
        class_to_players[cid] = tuple(idxs)
        class_order.append(cid)

    if total_players > 0:
        n_present_players = len(concept_list) - n_bg
        n_pad = total_players - n_present_players
        if n_pad < 0:
            raise SkipImage(
                f"n_present_players={n_present_players} exceeds --total-players={total_players}")
        if n_pad > 0:
            present_set = {int(c) for c in present_class_ids}
            candidates = [int(t[0]) for t in ctx.things_kept
                          if int(t[3]) == 1 and int(t[0]) not in present_set]
            rng = random.Random(f"distractor-{image_id}")
            sampled = rng.sample(candidates, min(n_pad, len(candidates)))
            for did in sampled:
                for tok in ctx.class_id_to_tokens[did]:
                    concept_list.append(tok)

    player_idx = tuple(range(n_bg, len(concept_list)))
    return concept_list, player_idx, context_idx, class_to_players, class_order


def run_ca(pipeline, image, concepts, player_idx, context_idx,
           layer_indices, width, height, seed, num_steps,
           noise_timestep, prompt, device, joint_attention_kwargs=None):
    """Paper CA: one `softmax=True` forward pass returning per-concept H×W maps
    where each map is mean_{t,l}( softmax_C(raw[t,l,:]) )[c]."""
    full_idx = sorted(player_idx + context_idx)
    full_concepts = [concepts[i] for i in full_idx]
    out = pipeline.encode_image(
        image=image, concepts=full_concepts, prompt=prompt,
        layer_indices=list(layer_indices),
        width=width, height=height, seed=seed,
        num_steps=num_steps, noise_timestep=noise_timestep,
        device=device, return_pil_heatmaps=False,
        softmax=True, joint_attention_kwargs=joint_attention_kwargs,
    )
    arr = np.asarray(out.concept_heatmaps, dtype=np.float32)
    assert arr.ndim == 3 and arr.shape[0] == len(full_idx), \
        f"unexpected CA shape {arr.shape} for {len(full_idx)} concepts"
    return {abs_idx: arr[k] for k, abs_idx in enumerate(full_idx)}


def run_meta_ca(pipeline, image, concepts, player_idx, context_idx,
                target_players, layer_indices, width, height, seed,
                num_steps, noise_timestep, prompt, device, chunk_size=10000,
                joint_attention_kwargs=None):
    """One `softmax=False` forward pass, then a streaming pass over coalition
    sizes 1..N (batched per size, GPU softmax per chunk) computes the Shapley
    diagonal sv[t, t] for each t in target_players. Constant memory wrt 2^N.

    Memory bound per chunk of `chunk_size` size-k coalitions:
        chunk_size · (k + |context|) · H · W · 4 bytes  (the softmax tensor)
    Peak k = floor(N/2). At N=20, chunk_size=10000 → ~5 GB peak.

    Returns (sv_diag_dict, H, W).
    """
    import torch

    full_idx = sorted(player_idx + context_idx)
    full_concepts = [concepts[i] for i in full_idx]
    out = pipeline.encode_image(
        image=image, concepts=full_concepts, prompt=prompt,
        layer_indices=list(layer_indices),
        width=width, height=height, seed=seed,
        num_steps=num_steps, noise_timestep=noise_timestep,
        device=device, return_pil_heatmaps=False,
        softmax=False, joint_attention_kwargs=joint_attention_kwargs,
    )
    raw_np = np.asarray(out.concept_heatmaps, dtype=np.float32)
    assert raw_np.ndim == 3 and raw_np.shape[0] == len(full_idx), \
        f"unexpected raw shape {raw_np.shape} for {len(full_idx)} concepts"
    H, W = raw_np.shape[1:]
    full_idx_arr = np.array(full_idx, dtype=np.int64)
    raw = torch.from_numpy(raw_np).to(device)

    n = len(player_idx)
    sv_diag_t = {p: torch.zeros((H, W), dtype=torch.float32, device=device)
                 for p in target_players}
    context_arr = np.array(sorted(context_idx), dtype=np.int64)

    for k in range(1, n + 1):
        coalitions = list(itertools.combinations(player_idx, k))
        if not coalitions:
            continue
        weight = (math.factorial(k - 1) * math.factorial(n - k)
                  / math.factorial(n))

        for ck in range(0, len(coalitions), chunk_size):
            chunk = coalitions[ck:ck + chunk_size]
            n_chunk = len(chunk)
            T_arr = np.array(chunk, dtype=np.int64)
            if context_arr.size > 0:
                ctx_block = np.broadcast_to(
                    context_arr, (n_chunk, context_arr.size))
                members = np.concatenate([T_arr, ctx_block], axis=1)
            else:
                members = T_arr
            local = np.searchsorted(full_idx_arr, members)
            local_t = torch.from_numpy(local).to(device)
            batch = raw[local_t]
            soft = torch.softmax(batch, dim=1)
            del batch

            for t in target_players:
                t_mask_np = (T_arr == t).any(axis=1)
                if not t_mask_np.any():
                    continue
                positions_np = (T_arr[t_mask_np] == t).argmax(axis=1).astype(np.int64)
                t_mask = torch.from_numpy(t_mask_np).to(device)
                positions = torch.from_numpy(positions_np).to(device)
                soft_with_t = soft[t_mask]
                idx = positions.view(-1, 1, 1, 1).expand(
                    -1, 1, soft_with_t.size(2), soft_with_t.size(3))
                t_softmax = torch.gather(soft_with_t, dim=1, index=idx).squeeze(1)
                sv_diag_t[t] = sv_diag_t[t] + weight * t_softmax.sum(dim=0)
                del soft_with_t, t_softmax
            del soft, local_t

    sv_diag = {p: v.cpu().numpy().astype(np.float32) for p, v in sv_diag_t.items()}
    return sv_diag, H, W


def class_channel_from_player_dict(channel_dict, class_to_players):
    out = {}
    for cid, players in class_to_players.items():
        if len(players) != 1:
            raise ValueError(
                f"class {cid} has {len(players)} players; expected 1 (single-token only)")
        out[cid] = channel_dict[players[0]]
    return out


# =============================================================================
# Per-image counts
# =============================================================================

def _resize_nearest_int(arr_int, size_hw):
    H, W = size_hw
    img = PIL.Image.fromarray(arr_int.astype(np.int32), mode="I")
    img = img.resize((W, H), resample=PIL.Image.NEAREST)
    return np.asarray(img, dtype=np.int64)


def _resize_nearest_float(arr_float, size_hw):
    H, W = size_hw
    img = PIL.Image.fromarray(arr_float.astype(np.float32), mode="F")
    img = img.resize((W, H), resample=PIL.Image.NEAREST)
    return np.asarray(img, dtype=np.float32)


def stack_and_argmax(channels_by_class, bg_cat_ids, full_cache_entry,
                     context_idx, class_order):
    """Per-pixel argmax over [bg channels, thing channels] → cat_id."""
    bg_chans = [full_cache_entry[i] for i in context_idx]
    thing_chans = [channels_by_class[cid] for cid in class_order]
    stack = np.stack(bg_chans + thing_chans)
    arg = stack.argmax(0)
    pred = np.zeros_like(arg, dtype=np.int64)
    n_bg = len(bg_chans)
    for k, cid in enumerate(bg_cat_ids):
        pred[arg == k] = int(cid)
    for k, cid in enumerate(class_order):
        pred[arg == (n_bg + k)] = int(cid)
    return pred


def per_image_counts_argmax(pred_id, target_id, classes_to_score):
    """Pooled per-class (inter, union, count) for the multi (argmax) track.
    target_id == 0 → unlabeled (excluded from per-class union)."""
    labeled = target_id > 0
    per_class = {}
    for c in classes_to_score:
        c = int(c)
        pred_c = (pred_id == c) & labeled
        targ_c = (target_id == c) & labeled
        per_class[str(c)] = (int((pred_c & targ_c).sum()),
                             int((pred_c | targ_c).sum()),
                             int(targ_c.sum()))
    return per_class


def per_image_counts_binary(saliency_hi, target_hi, cat_id):
    """Single-track protocol: min-max normalize → threshold at mean → 2-class
    IoU/Acc/AP for (saliency channel, cat_id)."""
    valid = target_hi != 0
    n_labeled = int(valid.sum())
    if n_labeled == 0:
        return {"correct": 0, "labeled": 0,
                "inter_fg": 0, "union_fg": 0, "inter_bg": 0, "union_bg": 0,
                "ap": float("nan")}
    target_bin = (target_hi == cat_id).astype(np.int64)
    arr = saliency_hi.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    sal_n = (arr - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(arr)
    pred_bin = (sal_n > sal_n.mean()).astype(np.int64)
    pv = pred_bin[valid]
    tv = target_bin[valid]
    inter_fg = int(((pv == 1) & (tv == 1)).sum())
    union_fg = int(((pv == 1) | (tv == 1)).sum())
    inter_bg = int(((pv == 0) & (tv == 0)).sum())
    union_bg = int(((pv == 0) | (tv == 0)).sum())
    correct = int((pv == tv).sum())

    sal_v = sal_n[valid].ravel()
    tgt_v = tv.ravel()
    pred_flat = np.concatenate([1.0 - sal_v, sal_v])
    tgt_flat = np.concatenate([(tgt_v == 0).astype(int),
                               (tgt_v == 1).astype(int)])
    if tgt_flat.sum() == 0 or tgt_flat.sum() == tgt_flat.size:
        ap = float("nan")
    else:
        try:
            ap = float(np.nan_to_num(average_precision_score(tgt_flat, pred_flat)))
        except ValueError:
            ap = float("nan")
    return {"correct": correct, "labeled": n_labeled,
            "inter_fg": inter_fg, "union_fg": union_fg,
            "inter_bg": inter_bg, "union_bg": union_bg, "ap": ap}


def per_image_ap(channels_hi, target_hi, present_class_ids):
    labeled = target_hi != 0
    out = {}
    for c in present_class_ids:
        c = int(c)
        chan = channels_hi.get(c)
        if chan is None:
            continue
        y_true = ((target_hi == c) & labeled).reshape(-1).astype(np.int8)
        y_score = chan.reshape(-1).astype(np.float32)[labeled.reshape(-1)]
        y_true = y_true[labeled.reshape(-1)]
        if y_true.sum() == 0 or y_true.sum() == y_true.size:
            continue
        try:
            ap = float(np.nan_to_num(average_precision_score(y_true, y_score)))
        except ValueError:
            ap = float("nan")
        out[str(c)] = ap
    return out


# =============================================================================
# Tracks + aggregators
# =============================================================================

def define_tracks(dataset_name, things_kept):
    """`single` is the binary-threshold track on 1-thing-class images.
    `multi` is the argmax track on all images (skipped for ImageNet-Seg whose
    n_GT=1 makes it equivalent to the single track)."""
    single_token_ids = {int(t[0]) for t in things_kept if int(t[3]) == 1}

    def _present(r):
        return r.get("present_class_ids") or []

    tracks = {
        "single": {"image_filter": lambda r: len(_present(r)) == 1,
                   "score_classes": single_token_ids, "protocol": "binary"},
    }
    if dataset_name != "imagenetseg":
        tracks["multi"] = {"image_filter": lambda r: True,
                           "score_classes": single_token_ids,
                           "protocol": "argmax"}
    return tracks


def aggregate_argmax_track(rows, key, score_classes):
    """Multi (argmax) protocol. Per-class IoU pooled over score_classes, then
    averaged → mIoU. mAP = mean over classes of (mean per-image AP). mAcc
    follows paper §D.3: numerator = pooled correct predictions where GT is an
    in-vocabulary thing class (= sum of per-class TP); denominator = total
    in-vocabulary thing-pixel count."""
    inter_pc, union_pc = {}, {}
    count_pc = {}
    aps_pc = {}
    for r in rows:
        if key not in r:
            continue
        s = r[key]
        for c_str, tup in s["per_class"].items():
            c = int(c_str)
            if c not in score_classes:
                continue
            inter_pc[c] = inter_pc.get(c, 0) + tup[0]
            union_pc[c] = union_pc.get(c, 0) + tup[1]
            if len(tup) >= 3:
                count_pc[c] = count_pc.get(c, 0) + tup[2]
        for c_str, ap in s.get("ap_per_class", {}).items():
            c = int(c_str)
            if c not in score_classes or ap is None:
                continue
            if isinstance(ap, float) and np.isnan(ap):
                continue
            aps_pc.setdefault(c, []).append(float(ap))
    eps = 1e-12
    iou_pc = {c: inter_pc[c] / (union_pc[c] + eps)
              for c in inter_pc if union_pc[c] > 0}
    miou = float(np.mean(list(iou_pc.values()))) if iou_pc else float("nan")
    ap_pc = {c: float(np.mean(v)) for c, v in aps_pc.items() if v}
    mAP  = float(np.mean(list(ap_pc.values()))) if ap_pc else float("nan")
    thing_correct = sum(inter_pc.values())
    thing_labeled = sum(count_pc.values())
    mAcc = (thing_correct / (thing_labeled + eps)
            if thing_labeled else float("nan"))
    return {"n_images": len(rows), "n_classes_with_iou": len(iou_pc),
            "mAcc": mAcc, "mIoU": miou, "mAP": mAP,
            "iou_per_class": {int(c): float(v) for c, v in iou_pc.items()},
            "ap_per_class":  {int(c): float(v) for c, v in ap_pc.items()},
            "protocol": "argmax"}


def aggregate_binary_track(rows, key, score_classes):
    """Single (binary) protocol. Each (image, in-vocab thing cat_id) pair scored
    as 'this class vs everything else'. Acc = pooled correct/labeled; mIoU =
    ½(fg_IoU + bg_IoU) pooled across observations; mAP = mean per-image AP."""
    inter_fg = union_fg = inter_bg = union_bg = 0
    correct = labeled = 0
    aps = []
    n_obs = 0
    iou_per_class_acc = {}
    for r in rows:
        if key not in r:
            continue
        for c_str, counts in r[key].get("binary_per_class", {}).items():
            c = int(c_str)
            if c not in score_classes:
                continue
            n_obs += 1
            inter_fg += counts["inter_fg"]; union_fg += counts["union_fg"]
            inter_bg += counts["inter_bg"]; union_bg += counts["union_bg"]
            correct  += counts["correct"];  labeled  += counts["labeled"]
            ap = counts.get("ap")
            if ap is not None and not (isinstance(ap, float) and np.isnan(ap)):
                aps.append(float(ap))
            agg = iou_per_class_acc.setdefault(c, [0, 0, 0, 0, []])
            agg[0] += counts["inter_fg"]; agg[1] += counts["union_fg"]
            agg[2] += counts["inter_bg"]; agg[3] += counts["union_bg"]
            if ap is not None and not (isinstance(ap, float) and np.isnan(ap)):
                agg[4].append(float(ap))
    eps = 1e-12
    iou_fg = inter_fg / (union_fg + eps) if union_fg else float("nan")
    iou_bg = inter_bg / (union_bg + eps) if union_bg else float("nan")
    miou = 0.5 * (iou_fg + iou_bg) if (union_fg and union_bg) else float("nan")
    iou_per_class = {}
    ap_per_class = {}
    for c, agg in iou_per_class_acc.items():
        i_fg, u_fg, i_bg, u_bg, ap_list = agg
        if u_fg and u_bg:
            iou_per_class[int(c)] = 0.5 * (i_fg / (u_fg + eps) + i_bg / (u_bg + eps))
        if ap_list:
            ap_per_class[int(c)] = float(np.mean(ap_list))
    return {"n_images": len(rows), "n_obs": n_obs,
            "mAcc": correct / (labeled + eps) if labeled else float("nan"),
            "mIoU": miou, "mAP": float(np.mean(aps)) if aps else float("nan"),
            "iou_fg": iou_fg, "iou_bg": iou_bg,
            "iou_per_class": iou_per_class, "ap_per_class": ap_per_class,
            "protocol": "binary"}


def aggregate_track(rows, key, track_spec):
    filt = [r for r in rows if track_spec["image_filter"](r)]
    sc = track_spec["score_classes"]
    if track_spec["protocol"] == "binary":
        return aggregate_binary_track(filt, key, sc)
    return aggregate_argmax_track(filt, key, sc)


# =============================================================================
# Per-image runner
# =============================================================================

def run_one_image(pipeline, ctx, entry, args):
    img = PIL.Image.open(entry["jpg_path"]).convert("RGB")
    target_native = ctx.decode_target(entry)
    if args.n_distractors is not None:
        effective_total = len(entry["present_class_ids"]) + args.n_distractors
    else:
        effective_total = args.total_players
    concepts, player_idx, context_idx, class_to_players, class_order = \
        build_player_layout(entry["present_class_ids"], ctx,
                            total_players=effective_total,
                            image_id=entry["image_id"])

    if args.prompt:
        prompt = ",".join(f"a {ctx.class_id_to_short[cid]}"
                          for cid in entry["present_class_ids"])
    else:
        prompt = ""

    jak = ({"concept_cross_attention": True, "concept_self_attention": False}
           if args.decoupled else None)
    ca_full = run_ca(
        pipeline, img, concepts, player_idx, context_idx,
        layer_indices=args.layer_indices, width=args.width, height=args.height,
        seed=args.seed, num_steps=args.num_steps,
        noise_timestep=args.noise_timestep, prompt=prompt,
        device=args.device, joint_attention_kwargs=jak,
    )
    target_players = tuple(class_to_players[cid][0] for cid in class_order)
    sv_diag, H, W = run_meta_ca(
        pipeline, img, concepts, player_idx, context_idx,
        target_players=target_players,
        layer_indices=args.layer_indices, width=args.width, height=args.height,
        seed=args.seed, num_steps=args.num_steps,
        noise_timestep=args.noise_timestep, prompt=prompt,
        device=args.device, chunk_size=args.chunk_size,
        joint_attention_kwargs=jak,
    )

    # Meta-CA argmax track reuses CA's bg-channel softmax — both methods share
    # the same in-coalition softmax for the always-present context concepts.
    meta_ca_channels = {cid: sv_diag[class_to_players[cid][0]] for cid in class_order}
    method_inputs = {
        "ca":      (class_channel_from_player_dict(ca_full, class_to_players), ca_full),
        "meta_ca": (meta_ca_channels, ca_full),
    }
    target_hi = _resize_nearest_int(target_native.astype(np.int64), EVAL_RES)
    classes_to_score = list(set(ctx.bg_cat_ids)) + list(class_order)

    out_metrics = {}
    for method_key, (channels_by_class, full_entry) in method_inputs.items():
        pred = stack_and_argmax(channels_by_class, ctx.bg_cat_ids,
                                full_entry, context_idx, class_order)
        pred_hi = _resize_nearest_int(pred, EVAL_RES)
        pc_iou = per_image_counts_argmax(pred_hi, target_hi, classes_to_score)
        chans_hi = {cid: _resize_nearest_float(c, EVAL_RES)
                    for cid, c in channels_by_class.items()}
        ap = per_image_ap(chans_hi, target_hi, class_order)
        binary_pc = {str(int(cid)): per_image_counts_binary(
                        chans_hi[cid], target_hi, int(cid))
                     for cid in class_order}
        out_metrics[method_key] = {"per_class": pc_iou, "ap_per_class": ap,
                                   "binary_per_class": binary_pc}
    return {"image_id": entry["image_id"],
            "present_class_ids": [int(c) for c in entry["present_class_ids"]],
            "n_present": len(class_order),
            "n_players": len(player_idx),
            **out_metrics}


# =============================================================================
# CLI + run_inference + run_aggregate
# =============================================================================

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()


def run_inference(args):
    log_path = os.path.join(args.output_dir,
                            f"shard_{args.shard_i}_of_{args.shard_n}.log")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print(f"=== Run started: host={socket.gethostname()}  pid={os.getpid()}"
          f"  dataset={args.dataset}  shard {args.shard_i}/{args.shard_n} ===",
          flush=True)
    print(f"  layer_indices={args.layer_indices}", flush=True)
    print(f"  bg_concepts={BG_CONCEPTS}", flush=True)

    import torch
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "12"))
    n_threads = max(1, slurm_cpus - 2)
    torch.set_num_threads(n_threads)
    print(f"  torch.set_num_threads({n_threads})", flush=True)

    set_seeds(args.seed)
    print(f"  set_seeds(seed={args.seed})", flush=True)

    print("loading T5 tokenizer…", flush=True)
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    loader = {"mscoco": load_mscoco, "pascalvoc": load_pascalvoc,
              "imagenetseg": load_imagenetseg}[args.dataset]
    ctx = loader(args, tokenizer)

    eligible = ctx.eligible
    print(f"  eligible images: {len(eligible)}", flush=True)
    k_hist = Counter(len(e["present_class_ids"]) for e in eligible)
    print(f"  k (= # present thing classes) distribution: "
          f"{dict(sorted(k_hist.items()))}", flush=True)
    if args.max_images is not None:
        eligible = eligible[: args.max_images]
        print(f"  after --max-images={args.max_images}: {len(eligible)} kept",
              flush=True)
    my_targets = [e for i, e in enumerate(eligible)
                  if i % args.shard_n == args.shard_i]
    print(f"  shard owns {len(my_targets)} images", flush=True)
    if not my_targets:
        sys.exit("nothing in this shard")

    if ConceptAttentionFluxPipeline is None:
        sys.exit("concept_attention not importable — use the metaconceptattention env.")
    print("loading ConceptAttentionFluxPipeline…", flush=True)
    t_pipe = time.perf_counter()
    pipeline = ConceptAttentionFluxPipeline(model_name="flux-schnell",
                                            device=args.device)
    print(f"  pipeline ready in {time.perf_counter() - t_pipe:.1f}s", flush=True)

    rows = []
    n_done = n_failed = n_skipped = 0
    overall_start = time.perf_counter()
    for entry in tqdm(my_targets, desc="images"):
        t0 = time.perf_counter()
        try:
            row = run_one_image(pipeline, ctx, entry, args)
            rows.append(row)
            tqdm.write(f">>> img {entry['image_id']} "
                       f"(k={row['n_present']}, N={row['n_players']})  "
                       f"wall={time.perf_counter()-t0:.1f}s")
            n_done += 1
        except SkipImage as e:
            tqdm.write(f"... img {entry['image_id']} skipped: {e}")
            n_skipped += 1
        except Exception as e:
            tqdm.write(f"!!! img {entry['image_id']} FAILED: {e!r}")
            import traceback; traceback.print_exc()
            n_failed += 1

    out_path = os.path.join(args.output_dir,
                            f"shard_{args.shard_i}_of_{args.shard_n}.json")
    with open(out_path, "w") as f:
        json.dump({"shard_i": args.shard_i, "shard_n": args.shard_n,
                   "dataset": args.dataset,
                   "layer_indices": list(args.layer_indices),
                   "bg_concepts": list(ctx.bg_concepts),
                   "bg_cat_ids": [int(c) for c in ctx.bg_cat_ids],
                   "total_players": int(args.total_players),
                   "n_distractors": (None if args.n_distractors is None
                                     else int(args.n_distractors)),
                   "prompt": bool(args.prompt),
                   "decoupled": bool(args.decoupled),
                   "n_done": n_done, "n_failed": n_failed, "n_skipped": n_skipped,
                   "things_kept": [list(t) for t in ctx.things_kept],
                   "stuff_kept":  [list(s) for s in ctx.stuff_kept],
                   "rows": rows}, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)
    print(f"=== shard done in {time.perf_counter() - overall_start:.1f}s "
          f"({n_done} ok, {n_skipped} skipped, {n_failed} failed) ===", flush=True)


def run_aggregate(args):
    shards = sorted(glob(os.path.join(args.output_dir, "shard_*_of_*.json")))
    if not shards:
        sys.exit(f"no shard JSONs under {args.output_dir}")
    print(f"aggregating {len(shards)} shards", flush=True)

    all_rows = []
    things_kept = stuff_kept = bg_concepts = layer_indices = None
    for s in shards:
        with open(s) as f:
            data = json.load(f)
        all_rows.extend(data["rows"])
        if things_kept is None:
            things_kept   = data["things_kept"]
            stuff_kept    = data["stuff_kept"]
            bg_concepts   = data.get("bg_concepts", [])
            layer_indices = data.get("layer_indices", DEFAULT_LAYERS)

    tracks = define_tracks(args.dataset, things_kept)
    paper_tracks = {k: {tname: aggregate_track(all_rows, k, tspec)
                        for tname, tspec in tracks.items()}
                    for k in METHOD_KEYS}
    track_counts = {tname: sum(1 for r in all_rows if tspec["image_filter"](r))
                    for tname, tspec in tracks.items()}
    for k in METHOD_KEYS:
        for tname, cell in paper_tracks[k].items():
            for sub in ("iou_per_class", "ap_per_class"):
                if sub in cell:
                    cell[sub] = {str(c): v for c, v in cell[sub].items()}

    id_to_short = {str(t[0]): t[2] for t in things_kept}
    id_to_short.update({str(s[0]): s[2] for s in stuff_kept})

    metrics = {
        "dataset": args.dataset,
        "n_images": len(all_rows),
        "n_shards": len(shards),
        "layer_indices": layer_indices,
        "bg_concepts": bg_concepts,
        "track_counts": track_counts,
        "paper_tracks": paper_tracks,
        "class_id_to_short": id_to_short,
    }
    out_json = os.path.join(args.output_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {out_json}", flush=True)

    print(f"\n=== overall (n={len(all_rows)} images) ===", flush=True)
    print(f"  tracks  {track_counts}", flush=True)
    width = max(len(METHOD_LABELS[k]) for k in METHOD_KEYS)
    for k in METHOD_KEYS:
        for tname, cell in paper_tracks[k].items():
            print(f"  {METHOD_LABELS[k]:<{width}s}  {tname:<6s}  "
                  f"[{cell['protocol']:<6s}]  "
                  f"Acc={cell['mAcc']:.4f}  mIoU={cell['mIoU']:.4f}  "
                  f"mAP={cell['mAP']:.4f}  (n={cell['n_images']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("mscoco", "pascalvoc", "imagenetseg"),
                    default="mscoco")
    ap.add_argument("--output-dir", default=None,
                    help=f"Default: {RESULTS_BASE}/<dataset>/")
    ap.add_argument("--shard-i", type=int, default=0)
    ap.add_argument("--shard-n", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=None,
                    help="Cap eligible image count.")
    ap.add_argument("--aggregate", action="store_true",
                    help="Skip inference; pool shards into metrics.json.")
    ap.add_argument("--layer-indices", type=int, nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--total-players", type=int, default=20,
                    help="Pad each image's player set to exactly this many "
                         "single-token thing classes (GT-present + random "
                         "distractors). 0 disables padding.")
    ap.add_argument("--n-distractors", type=int, default=None,
                    help="Variable-size: per-image total = n_present + N. "
                         "Overrides --total-players when set.")
    ap.add_argument("--prompt", action="store_true",
                    help="Feed FLUX an artificial 'a {cls1}, a {cls2}, …' "
                         "prompt built from GT-present classes (paper Fig 13 "
                         "'with artificial prompt' ablation). Default: empty.")
    ap.add_argument("--decoupled", action="store_true",
                    help="concept_self_attention=False on both forward passes "
                         "(paper Fig 13 'without cross-concept attention').")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--noise-timestep", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=10000,
                    help="Coalitions per GPU softmax call (memory knob).")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.width != args.height:
        ap.error("--width must equal --height (flux-schnell asserts square inputs)")
    if not (0 <= args.shard_i < args.shard_n):
        ap.error(f"--shard-i must be in [0, {args.shard_n})")
    if args.output_dir is None:
        args.output_dir = os.path.join(RESULTS_BASE, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.aggregate:
        return run_aggregate(args)
    return run_inference(args)


if __name__ == "__main__":
    main()
