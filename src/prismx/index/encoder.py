"""PRISMX Dense Encoder with Resumable Off-OneDrive Checkpointing and Float32 Artifact Persistence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OFF_ONEDRIVE_STORAGE = Path("C:/Users/manis/AppData/Local/prismx_storage")

class DenseEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_seq_length: int = 128,
        embedding_dim: int = 384,
        torch_threads: int = 12,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.torch_threads = torch_threads
        self.device = device
        self.cache_dir = cache_dir or (OFF_ONEDRIVE_STORAGE / "embedding_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        torch.set_num_threads(torch_threads)
        print(f"Loading SentenceTransformer '{self.model_name}' on {device} (torch threads={torch_threads})...")
        t0 = time.time()
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.max_seq_length = self.max_seq_length
        self.load_time = time.time() - t0
        print(f"Model loaded in {self.load_time:.2f}s.")

    def encode_queries(self, queries: list[str] | str, prefix: str = "") -> np.ndarray:
        """Encodes queries with optional BGE retrieval instruction prefix."""
        if isinstance(queries, str):
            queries = [queries]
        if prefix:
            queries = [f"{prefix}{q}" for q in queries]
        return self.model.encode(
            queries,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def encode_corpus_checkpointed(
        self,
        passages: list[dict[str, str]],
        batch_size: int = 256,
        checkpoint_every: int = 5000,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """Encodes a full passage corpus with resumable disk checkpoints off OneDrive.
        
        Saves raw normalized embeddings as float32 .npy and returns the array, ordered IDs, and timing metrics.
        """
        n_passages = len(passages)
        ordered_ids = [str(p["passage_id"]) for p in passages]
        texts = [p["text"] for p in passages]

        final_npy_path = self.cache_dir / f"embeddings_{n_passages}_dim{self.embedding_dim}.npy"
        final_ids_path = self.cache_dir / f"passage_ids_{n_passages}.json"

        # Check if completed artifact already exists
        if final_npy_path.exists() and final_ids_path.exists():
            print(f"Found completed embedding artifact at {final_npy_path}. Loading...")
            embeddings = np.load(final_npy_path)
            with open(final_ids_path, "r", encoding="utf-8") as f:
                saved_ids = json.load(f)
            if len(embeddings) == n_passages and saved_ids == ordered_ids:
                sha = hashlib.sha256(final_npy_path.read_bytes()).hexdigest()
                return embeddings, ordered_ids, {
                    "from_cache": True,
                    "encode_time_s": 0.0,
                    "passages_per_sec": 0.0,
                    "sha256": sha,
                    "path": str(final_npy_path),
                }

        print(f"Encoding {n_passages} passages in batches of {batch_size} (resumable checkpoints every {checkpoint_every})...")
        t0 = time.time()
        all_embeddings = np.zeros((n_passages, self.embedding_dim), dtype=np.float32)

        # Check for existing shard checkpoints
        completed_up_to = 0
        shard_idx = 0
        while True:
            shard_file = self.cache_dir / f"shard_{shard_idx}_{checkpoint_every}.npy"
            if shard_file.exists():
                shard_data = np.load(shard_file)
                shard_len = len(shard_data)
                all_embeddings[completed_up_to : completed_up_to + shard_len] = shard_data
                completed_up_to += shard_len
                shard_idx += 1
            else:
                break

        if completed_up_to > 0:
            print(f"Resuming encoding from passage {completed_up_to:,} / {n_passages:,}...")

        # Encode remaining
        curr_shard = []
        for i in range(completed_up_to, n_passages, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_emb = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            all_embeddings[i : i + len(batch_texts)] = batch_emb
            curr_shard.append(batch_emb)

            current_count = i + len(batch_texts)
            if (current_count - completed_up_to) >= checkpoint_every or current_count == n_passages:
                shard_arr = np.vstack(curr_shard)
                shard_file = self.cache_dir / f"shard_{shard_idx}_{checkpoint_every}.npy"
                np.save(shard_file, shard_arr)
                completed_up_to = current_count
                shard_idx += 1
                curr_shard = []
                elapsed = time.time() - t0
                speed = current_count / elapsed if elapsed > 0 else 0
                print(f"  Encoded {current_count:,} / {n_passages:,} passages ({speed:.1f} pass/sec)...")

        total_encode_time = time.time() - t0
        passages_per_sec = n_passages / total_encode_time if total_encode_time > 0 else 0

        # Save consolidated artifact atomically
        temp_npy = self.cache_dir / "temp_embeddings.npy"
        np.save(temp_npy, all_embeddings)
        if final_npy_path.exists():
            final_npy_path.unlink()
        temp_npy.rename(final_npy_path)

        with open(final_ids_path, "w", encoding="utf-8") as f:
            json.dump(ordered_ids, f)

        # Clean shard checkpoints
        for f in self.cache_dir.glob("shard_*.npy"):
            try:
                f.unlink()
            except Exception:
                pass

        sha = hashlib.sha256(final_npy_path.read_bytes()).hexdigest()
        print(f"Saved consolidated embeddings: {final_npy_path} (SHA-256: {sha[:16]}...)")

        meta = {
            "from_cache": False,
            "encode_time_s": round(total_encode_time, 2),
            "passages_per_sec": round(passages_per_sec, 1),
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "torch_threads": self.torch_threads,
            "batch_size": batch_size,
            "sha256": sha,
            "path": str(final_npy_path),
        }
        return all_embeddings, ordered_ids, meta
