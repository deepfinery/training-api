"""Utilities for downloading datasets and materializing local shards."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import boto3


def download_dataset(dataset_uri: str, run_id: str) -> Path:
    """Download the dataset (currently S3 or local path) and return a local path."""

    parsed = _parse_uri(dataset_uri)
    if parsed[0] == "file":
        return Path(parsed[2]).expanduser().resolve()

    if parsed[0] != "s3":
        raise ValueError(f"Unsupported dataset scheme: {parsed[0]}")

    bucket, key = parsed[1], parsed[2].lstrip("/")
    suffix = Path(key).suffix or ".jsonl"
    fd, tmp_path = tempfile.mkstemp(prefix=f"dataset-{run_id}-", suffix=suffix)
    os.close(fd)
    boto3.client("s3").download_file(bucket, key, tmp_path)
    return Path(tmp_path)


def stream_jsonl(path: Path, text_field: str = "text") -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_field not in obj:
                raise KeyError(f"Dataset row missing '{text_field}' field")
            yield obj[text_field]


def split_corpus(lines: Iterable[str], max_samples: int | None = None) -> Tuple[List[str], List[str]]:
    train: List[str] = []
    eval_: List[str] = []
    for idx, line in enumerate(lines):
        if max_samples and idx >= max_samples:
            break
        if idx % 10 == 0:
            eval_.append(line)
        else:
            train.append(line)
    if not train or not eval_:
        raise ValueError("Dataset too small. Need at least 10 rows for train/eval split.")
    return train, eval_


def _parse_uri(uri: str) -> Tuple[str, str, str]:
    if uri.startswith("s3://"):
        bucket, _, key = uri[5:].partition("/")
        return "s3", bucket, key
    if uri.startswith("file://"):
        return "file", "", uri[7:]
    return "file", "", uri
