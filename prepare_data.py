#!/usr/bin/env python3
"""Tokenize FineWeb-Edu into flat uint16 shards for training.

Why pre-tokenize: the training run is 20B tokens and the cluster job may be
requeued.  A flat ``train.bin`` lets ``train.py`` memory-map and index tokens in
O(1), so a restart resumes exactly where it stopped instead of re-tokenizing
the stream from the beginning.

Examples
--------
Cluster (local dataset copy)::

    python prepare_data.py \
        --source /home/transformers/datasets/karpathy--fineweb-edu-100b \
        --out ./data/fineweb_20b --tokens 20000000000 --workers 32

Through the HF mirror::

    HF_ENDPOINT=https://hf-mirror.com python prepare_data.py \
        --source karpathy/fineweb-edu-100b-shuffle \
        --out ./data/fineweb_20b --tokens 20000000000 --workers 32

Local smoke test (no HF, no torch)::

    python prepare_data.py --source notes.txt --out ./data/tiny \
        --tokens 100000 --val-tokens 1000 --workers 0
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

EOS_ID = 2  # Llama-2 </s>

_TOKENIZER = None
_TEXT_FIELD = "text"


def _init_worker(tokenizer_path: str, text_field: str):
    global _TOKENIZER, _TEXT_FIELD
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
    _TEXT_FIELD = text_field


def _encode_batch(texts):
    """Tokenize a batch of documents and append EOS to each of them."""
    ids = _TOKENIZER(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    flat = []
    for row in ids:
        flat.extend(row)
        flat.append(EOS_ID)
    return np.asarray(flat, dtype=np.uint16)


# ---------------------------------------------------------------------------
# data sources -> iterator of list[str]
# ---------------------------------------------------------------------------

def _iter_from_text_files(paths, batch_docs):
    batch = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if path.endswith(".jsonl"):
                    try:
                        line = json.loads(line).get(_TEXT_FIELD, "")
                    except json.JSONDecodeError:
                        continue
                    if not line:
                        continue
                batch.append(line)
                if len(batch) >= batch_docs:
                    yield batch
                    batch = []
    if batch:
        yield batch


def _iter_from_dataset(ds, batch_docs):
    batch = []
    for ex in ds:
        text = ex.get(_TEXT_FIELD)
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_docs:
            yield batch
            batch = []
    if batch:
        yield batch


def build_source(args):
    global _TEXT_FIELD
    _TEXT_FIELD = args.text_field

    src = args.source
    batch_docs = args.batch_docs

    if os.path.isfile(src) and (src.endswith(".txt") or src.endswith(".jsonl")):
        return _iter_from_text_files([src], batch_docs), f"file:{src}"

    if os.path.isdir(src):
        parquet = sorted(glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True))
        txts = sorted(glob.glob(os.path.join(src, "**", "*.txt"), recursive=True)) + \
            sorted(glob.glob(os.path.join(src, "**", "*.jsonl"), recursive=True))
        if parquet:
            from datasets import load_dataset

            print(f"[data] {len(parquet)} parquet files under {src}", flush=True)
            # Pass the explicit file list instead of data_dir: data_dir makes
            # `datasets` treat *every* file in the directory as parquet, and a
            # stray dataset_info.md / README then blows up with
            # "Parquet magic bytes not found in footer".
            ds = load_dataset("parquet", data_files={"train": parquet}, split="train", streaming=True)
            return _iter_from_dataset(ds, batch_docs), f"parquet:{src}"
        if txts:
            print(f"[data] {len(txts)} text files under {src}", flush=True)
            return _iter_from_text_files(txts, batch_docs), f"text:{src}"
        raise SystemExit(f"No .parquet/.txt/.jsonl found under {src}")

    # HuggingFace hub dataset id
    from datasets import load_dataset

    kwargs = dict(split=args.hf_split, streaming=True)
    if args.hf_config:
        kwargs["name"] = args.hf_config
    print(f"[data] streaming HF dataset {src} (split={args.hf_split})", flush=True)
    ds = load_dataset(src, **kwargs)
    return _iter_from_dataset(ds, batch_docs), f"hf:{src}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Tokenize a corpus into uint16 .bin shards")
    p.add_argument("--source", required=True,
                   help="HF dataset id, a directory (parquet/txt), or a .txt/.jsonl file")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--tokens", type=int, default=20_000_000_000)
    p.add_argument("--val-tokens", type=int, default=10_000_000)
    p.add_argument("--tokenizer", type=str, default="./tokenizer")
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--hf-config", type=str, default=None)
    p.add_argument("--hf-split", type=str, default="train")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch-docs", type=int, default=512)
    p.add_argument("--max-docs", type=int, default=None, help="stop after N documents (debug)")
    p.add_argument("--progress-every", type=int, default=100_000_000)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    train_path = os.path.join(args.out, "train.bin")
    val_path = os.path.join(args.out, "val.bin")
    meta_path = os.path.join(args.out, "meta.json")

    batches, src_desc = build_source(args)

    if args.workers > 0:
        pool = Pool(args.workers, initializer=_init_worker,
                    initargs=(args.tokenizer, args.text_field))
        results = pool.imap(_encode_batch, batches, chunksize=1)
    else:
        _init_worker(args.tokenizer, args.text_field)
        pool = None
        results = map(_encode_batch, batches)

    train_f = io.open(train_path, "wb")
    val_f = io.open(val_path, "wb")

    n_train = 0
    n_val = 0
    docs = 0
    next_report = args.progress_every
    t0 = time.time()

    train_done = False
    try:
        for arr in results:
            docs += 1
            pos = 0

            if not train_done:
                # Document boundaries are exactly the EOS positions, so the train
                # file can be cut on a whole-document edge.  Cutting anywhere else
                # would leave one document half in train and half in val.
                ends = np.flatnonzero(arr == EOS_ID) + 1
                if len(ends) == 0:
                    pos = len(arr)  # no complete document in this batch
                else:
                    budget = args.tokens - n_train
                    # number of complete documents that still fit
                    k = int(np.searchsorted(ends, budget, side="right"))
                    if k == 0:
                        if n_train == 0:
                            k = 1  # always keep at least one document in train
                        else:
                            train_done = True
                    if k > 0:
                        cut = int(ends[k - 1])
                        train_f.write(arr[:cut].tobytes())
                        n_train += cut
                        pos = cut
                        if k < len(ends) or n_train >= args.tokens:
                            train_done = True

            while pos < len(arr) and n_val < args.val_tokens:
                take = min(len(arr) - pos, args.val_tokens - n_val)
                val_f.write(arr[pos:pos + take].tobytes())
                n_val += take
                pos += take

            if train_done and n_val >= args.val_tokens:
                break
            if args.max_docs is not None and docs >= args.max_docs:
                break

            if n_train >= next_report:
                elapsed = time.time() - t0
                print(f"[tokenize] {n_train/1e9:.2f}B train / {n_val/1e6:.1f}M val tokens, "
                      f"{docs} docs, {n_train/max(elapsed,1e-9)/1e3:.0f}k tok/s", flush=True)
                next_report += args.progress_every
    finally:
        train_f.close()
        val_f.close()
        if pool is not None:
            pool.terminate()
            pool.join()

    meta = dict(
        source=src_desc,
        train_tokens=int(n_train),
        val_tokens=int(n_val),
        docs=int(docs),
        dtype="uint16",
        eos_id=EOS_ID,
        split="document-boundary",  # train = 前 N 个完整文档，val 从下一个文档开始
        tokenizer=args.tokenizer,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[done] train.bin {n_train:,} tokens ({n_train*2/1e9:.1f} GB), "
          f"val.bin {n_val:,} tokens -> {args.out}", flush=True)

    if not train_done and n_train < args.tokens:
        print(f"[warn] source exhausted: only {n_train:,} train tokens "
              f"(< {args.tokens:,}).", flush=True)
    if n_val < args.val_tokens:
        print(f"[warn] source exhausted: only {n_val:,} val tokens "
              f"(< {args.val_tokens:,}).", flush=True)
    if n_train < args.tokens and train_done:
        print(f"[info] train stopped on a document boundary, "
              f"{args.tokens - n_train:,} tokens short of the target.", flush=True)

    # All files are closed and fsynced at this point.  Exit explicitly instead
    # of returning: with `datasets` + pyarrow + a multiprocessing.Pool, Python's
    # interpreter teardown can abort with
    #   "terminate called without an active exception"  (SIGABRT, core dumped)
    # long after the work is done.  That turned a successful run into exit 134.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
