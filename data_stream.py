"""On-the-fly text streaming + tokenisation + packing.

训练时直接从 parquet 流式读文本、用**当前配置的 tokenizer** 现 tokenize、打包成
定长序列。相比"先 prepare_data.py 全量 tokenize 成 bin"，好处是：

  * 换 tokenizer 只要改配置、重启，不需要重新处理整个语料；
  * 不占 40 GB 磁盘，也不用等 10 分钟的预处理；
  * 每个 rank 只读自己的文件（files[rank::world]），天然无重复。

代价是每个 step 要多花一点 CPU 做 tokenize（实测单进程 ~1M tok/s，相对 GPU
的 ~100k tok/s 可以忽略）。

断点续跑：流的位置（第几个文件、文件内第几篇文档、打包缓冲里剩的 token）会写进
checkpoint。重启时按位置重新打开文件、跳过已读文档（只解码 parquet，不重新
tokenize），然后从缓冲继续，所以不会重复消耗算力。
"""

from __future__ import annotations

import glob
import os
from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# file discovery / sharding
# ---------------------------------------------------------------------------

def list_parquet_files(source: str) -> List[str]:
    """Return the sorted parquet files under ``source`` (a directory)."""
    if not os.path.isdir(source):
        raise SystemExit(
            f"--data-source 需要是一个存放 .parquet 的目录，收到: {source!r}\n"
            "（在线 tokenize 模式不支持 HF dataset id，请先镜像到本地）"
        )
    files = sorted(glob.glob(os.path.join(source, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"{source} 下没有找到任何 .parquet 文件")
    return files


def shard_files(
    files: Sequence[str],
    world_size: int,
    rank: int,
    val_files: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split files into (this rank's train files, the shared val files).

    The last ``val_files`` files are reserved for validation and never used for
    training; every rank sees the same validation documents.
    """
    val_files = max(0, int(val_files))
    if val_files >= len(files):
        raise SystemExit(f"val_files={val_files} 但总共只有 {len(files)} 个 parquet 文件")

    if val_files:
        train_pool = list(files[:-val_files])
        val_pool = list(files[-val_files:])
    else:
        train_pool = list(files)
        val_pool = []

    mine = train_pool[rank::world_size]
    if not mine:
        raise SystemExit(f"rank {rank} 分不到任何训练文件（world_size={world_size}）")
    return mine, val_pool


# ---------------------------------------------------------------------------
# validation tokens (materialised once at startup, ~20 MB for 10M tokens)
# ---------------------------------------------------------------------------

def build_val_tokens(
    val_files: Sequence[str],
    tokenizer,
    target_tokens: int,
    eos_id: int = 2,
    log=print,
) -> np.ndarray:
    """Tokenise a fixed slice of held-out documents into a uint16 array."""
    import pyarrow.parquet as pq

    buf: List[int] = []
    for path in val_files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["text"]):
            texts = [t for t in batch.column("text").to_pylist() if t]
            if not texts:
                continue
            for row in tokenizer(
                texts, add_special_tokens=False, return_attention_mask=False
            )["input_ids"]:
                buf.extend(row)
                buf.append(eos_id)
            if len(buf) >= target_tokens:
                out = np.asarray(buf[:target_tokens], dtype=np.uint16)
                log(f"[data] val: {len(out)} tokens from {len(val_files)} held-out files")
                return out
    out = np.asarray(buf, dtype=np.uint16)
    log(f"[data] val: only {len(out)} tokens (target {target_tokens})")
    return out


# ---------------------------------------------------------------------------
# the training stream
# ---------------------------------------------------------------------------

class PackedStream:
    """Streams documents, tokenises them and yields fixed-length token rows."""

    def __init__(
        self,
        files: Sequence[str],
        tokenizer,
        seq_len: int,
        eos_id: int = 2,
        batch_docs: int = 256,
    ):
        self.files = list(files)
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.eos_id = int(eos_id)
        self.batch_docs = int(batch_docs)

        self.pending: List[int] = []   # tokens not yet turned into a sequence
        self.file_idx = 0              # index into self.files
        self.docs_in_file = 0          # documents consumed from the current file
        self.docs_total = 0            # documents consumed overall (for logging)
        self.tokens_total = 0          # tokens produced overall

        self._pf = None
        self._batches = None
        self._row_buf: List[str] = []
        self._skip = 0                 # docs to skip when opening the next file

    # -- iteration ----------------------------------------------------------
    def _open_current_file(self) -> bool:
        """Open self.files[self.file_idx]; returns False when out of files."""
        import pyarrow.parquet as pq

        if self.file_idx >= len(self.files):
            return False
        self._pf = pq.ParquetFile(self.files[self.file_idx])
        self._batches = self._pf.iter_batches(batch_size=256, columns=["text"])
        self._row_buf = []
        self.docs_in_file = 0

        # Fast-forward inside this file after a resume (decode only, no tokenise).
        to_skip = self._skip
        self._skip = 0
        while to_skip > 0:
            if not self._row_buf:
                try:
                    batch = next(self._batches)
                except StopIteration:
                    return False
                self._row_buf = [t for t in batch.column("text").to_pylist() if t]
                continue
            drop = min(to_skip, len(self._row_buf))
            del self._row_buf[:drop]
            to_skip -= drop
            self.docs_in_file += drop
        return True

    def _next_texts(self, n: int) -> List[str]:
        """Return up to ``n`` documents, crossing file boundaries as needed."""
        out: List[str] = []
        while len(out) < n:
            if self._batches is None:
                if not self._open_current_file():
                    break
            while len(out) < n and self._row_buf:
                out.append(self._row_buf.pop(0))
                self.docs_in_file += 1
            if len(out) >= n:
                break
            try:
                batch = next(self._batches)
            except StopIteration:
                # current file finished -> move on
                self._batches = None
                self._pf = None
                self.file_idx += 1
                continue
            self._row_buf = [t for t in batch.column("text").to_pylist() if t]
        self.docs_total += len(out)
        return out

    # -- public API ---------------------------------------------------------
    def next_tokens(self, n: int) -> np.ndarray:
        """Return the next ``n`` tokens, tokenising more documents as needed."""
        while len(self.pending) < n:
            texts = self._next_texts(self.batch_docs)
            if not texts:
                raise RuntimeError(
                    "训练数据流已耗尽：请减少 --tokens，或提供更多 parquet 文件"
                )
            encoded = self.tokenizer(
                texts, add_special_tokens=False, return_attention_mask=False
            )["input_ids"]
            for row in encoded:
                self.pending.extend(row)
                self.pending.append(self.eos_id)
        out = np.asarray(self.pending[:n], dtype=np.int64)
        del self.pending[:n]
        self.tokens_total += n
        return out

    def next_batch(self, micro: int, seq_len: int):
        """Return ``[micro, seq_len + 1]`` int64 token ids (x/y share this row)."""
        import torch

        flat = self.next_tokens(micro * (seq_len + 1))
        return torch.from_numpy(flat).view(micro, seq_len + 1)

    # -- checkpoint state ---------------------------------------------------
    def state_dict(self) -> dict:
        return dict(
            file_idx=int(self.file_idx),
            docs_in_file=int(self.docs_in_file),
            docs_total=int(self.docs_total),
            tokens_total=int(self.tokens_total),
            pending=[int(x) for x in self.pending],
        )

    def load_state_dict(self, state: dict) -> None:
        self.file_idx = int(state["file_idx"])
        self.docs_in_file = int(state.get("docs_in_file", 0))
        self.docs_total = int(state.get("docs_total", 0))
        self.tokens_total = int(state.get("tokens_total", 0))
        self.pending = [int(x) for x in state.get("pending", [])]
        # reopen at the saved position and skip already-consumed documents
        self._batches = None
        self._pf = None
        self._row_buf = []
        self._skip = self.docs_in_file

    def position(self) -> str:
        name = os.path.basename(self.files[self.file_idx]) if self.file_idx < len(self.files) else "-"
        return (f"file {self.file_idx}/{len(self.files)} ({name}) "
                f"doc {self.docs_in_file} total_docs {self.docs_total}")
