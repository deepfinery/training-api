"""torchrun entrypoint that dispatches to framework-specific trainers."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from frameworks import get_runner

logger = logging.getLogger("trainer")
STEP_PATTERN = re.compile(r"step[-=](\d+)")
HF_CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


def parse_args() -> tuple[argparse.Namespace, Dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Unified torchrun dispatcher")
    parser.add_argument("--framework", choices=["nemo", "hf", "meta"], default=os.getenv("FRAMEWORK"))
    parser.add_argument("--run-id", default=os.getenv("RUN_ID"))
    parser.add_argument("--checkpoint-base-uri", default=os.getenv("CHECKPOINT_BASE_URI"))
    parser.add_argument("--checkpoint-prefix", default=os.getenv("CHECKPOINT_PREFIX"))
    parser.add_argument("--dataset-uri", default=os.getenv("DATASET_URI"))
    parser.add_argument("--logs-uri", default=os.getenv("LOGS_URI"))
    parser.add_argument("--config-uri", default=os.getenv("CONFIG_URI"))
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID"))
    parser.add_argument("--config", dest="config_path", default=os.getenv("CONFIG_PATH"))
    parser.add_argument(
        "--resume",
        action="store_true",
        default=os.getenv("RESUME_FROM_CHECKPOINT") in {"1", "true", "True"},
        help="Resume from latest checkpoint",
    )
    parser.add_argument("--extra-args-json", default=os.getenv("TRAINING_EXTRA_ARGS_JSON"))

    known, unknown = parser.parse_known_args()
    required_fields = {
        "framework": "FRAMEWORK",
        "run_id": "RUN_ID",
        "checkpoint_base_uri": "CHECKPOINT_BASE_URI",
        "checkpoint_prefix": "CHECKPOINT_PREFIX",
        "dataset_uri": "DATASET_URI",
        "logs_uri": "LOGS_URI",
        "config_uri": "CONFIG_URI",
        "model_id": "MODEL_ID",
    }
    missing = [env for field, env in required_fields.items() if not getattr(known, field)]
    if missing:
        parser.error(f"Missing required flags or env vars: {', '.join(missing)}")
    overrides = parse_framework_cli(unknown)
    if known.extra_args_json:
        try:
            overrides.update(json.loads(known.extra_args_json))
        except json.JSONDecodeError:
            logger.warning("Invalid TRAINING_EXTRA_ARGS_JSON payload ignored")
    return known, overrides


def parse_framework_cli(cli: Iterable[str]) -> Dict[str, Any]:
    args = list(cli)
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token.lstrip("-").replace("-", "_")
        next_idx = i + 1
        if next_idx >= len(args) or args[next_idx].startswith("--"):
            overrides[key] = True
            i += 1
            continue
        overrides[key] = args[next_idx]
        i += 2
    return overrides


@dataclass
class StorageDriver:
    uri: str

    def __post_init__(self) -> None:
        parsed = urlparse(self.uri)
        scheme = parsed.scheme or "file"
        if scheme == "s3":
            if boto3 is None:
                raise RuntimeError("boto3 is required to work with S3 URIs")
            self.client = boto3.client("s3")
            self.bucket = parsed.netloc
            self.prefix = parsed.path.lstrip("/")
            if self.prefix.endswith("/"):
                self.prefix = self.prefix[:-1]
            self.kind = "s3"
            self.local_root = None
        else:
            path = parsed.path if scheme == "file" else self.uri
            self.local_root = Path(path or ".").expanduser().resolve()
            self.local_root.mkdir(parents=True, exist_ok=True)
            self.kind = "local"
            self.client = None
            self.prefix = ""

    def _full_key(self, relative: str | None = None) -> str:
        relative = (relative or "").lstrip("/")
        base = self.prefix.rstrip("/")
        if base and relative:
            return f"{base}/{relative}"
        if base:
            return base
        return relative

    def upload_bytes(self, relative: str, payload: bytes) -> None:
        relative = relative.strip("/")
        if self.kind == "s3":
            key = self._full_key(relative)
            self.client.put_object(Bucket=self.bucket, Key=key, Body=payload)
        else:
            assert self.local_root is not None
            dest = self.local_root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)

    def upload_json(self, relative: str, payload: Dict[str, Any]) -> None:
        self.upload_bytes(relative, json.dumps(payload, indent=2).encode("utf-8"))

    def upload_file(self, relative: str, file_path: Path) -> None:
        relative = relative.strip("/")
        if self.kind == "s3":
            key = self._full_key(relative)
            self.client.upload_file(str(file_path), self.bucket, key)
        else:
            assert self.local_root is not None
            dest = self.local_root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(file_path.read_bytes())

    def download_file(self, relative: str, destination: Path) -> None:
        relative = relative.strip("/")
        if self.kind == "s3":
            key = self._full_key(relative)
            self.client.download_file(self.bucket, key, str(destination))
        else:
            assert self.local_root is not None
            source = self.local_root / relative
            destination.write_bytes(source.read_bytes())

    def sync_directory(self, source: Path, remote_prefix: str) -> None:
        for path in source.rglob("*"):
            if path.is_file():
                rel = Path(remote_prefix.strip("/")) / path.relative_to(source)
                self.upload_file(rel.as_posix(), path)

    def download_prefix(self, prefix: str, destination: Path) -> None:
        prefix = prefix.strip("/")
        destination.mkdir(parents=True, exist_ok=True)
        if self.kind == "s3":
            paginator = self.client.get_paginator("list_objects_v2")
            full_prefix = self._full_key(prefix)
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = key[len(self.prefix) :].lstrip("/") if self.prefix else key
                    target = destination / Path(rel).relative_to(prefix)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    self.client.download_file(self.bucket, key, str(target))
        else:
            assert self.local_root is not None
            source = self.local_root / prefix
            for path in source.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(source)
                    dest = destination / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(path.read_bytes())

    def list_objects(self, relative_prefix: str) -> List[str]:
        prefix = relative_prefix.strip("/")
        if self.kind == "s3":
            full_prefix = self._full_key(prefix)
            paginator = self.client.get_paginator("list_objects_v2")
            objects: List[str] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if key.endswith("/"):
                        continue
                    rel = key[len(self.prefix) :].lstrip("/") if self.prefix else key
                    objects.append(rel)
            return objects
        assert self.local_root is not None
        base = self.local_root / prefix if prefix else self.local_root
        if not base.exists():
            return []
        results = []
        for path in base.rglob("*"):
            if path.is_file():
                results.append(str(path.relative_to(self.local_root)))
        return results


class CheckpointManager:
    def __init__(self, request: argparse.Namespace) -> None:
        base_uri = request.checkpoint_prefix or os.environ.get("CHECKPOINT_PREFIX")
        if not base_uri:
            raise ValueError("CHECKPOINT_PREFIX is required")
        self.driver = StorageDriver(base_uri)
        self.framework = request.framework
        self._local_root = Path(os.environ.get("TRAINING_LOCAL_ROOT", "/tmp/trainer")) / request.run_id
        self._local_root.mkdir(parents=True, exist_ok=True)

    def ensure_layout(self) -> None:
        for folder in ("checkpoints", "logs", "config"):
            if self.driver.kind == "local":
                self.driver.upload_bytes(f"{folder}/.keep", b"")
            else:
                try:
                    self.driver.upload_bytes(f"{folder}/.keep", b"")
                except (BotoCoreError, ClientError):
                    logger.debug("Skipping placeholder upload for %s", folder)

    def latest_checkpoint(self) -> Optional[str]:
        objects = self.driver.list_objects("checkpoints")
        latest_step = -1
        latest_path: Optional[str] = None
        for obj in objects:
            match = STEP_PATTERN.search(obj)
            dir_match = HF_CHECKPOINT_PATTERN.search(obj)
            if match:
                step = int(match.group(1))
                candidate = obj
            elif dir_match:
                step = int(dir_match.group(1))
                marker = dir_match.group(0)
                start = obj.rfind(marker)
                candidate = obj[: start + len(marker)] + "/"
            else:
                continue
            if step > latest_step:
                latest_step = step
                latest_path = candidate
        return latest_path

    def save_checkpoint(self, step: int, metadata: Dict[str, Any]) -> str:
        filename = f"checkpoints/step-{step:06d}.json"
        payload = {
            "framework": self.framework,
            "step": step,
            "saved_at": time.time(),
            "metadata": metadata,
        }
        self.driver.upload_json(filename, payload)
        return filename

    def persist_config(self, config_payload: Dict[str, Any]) -> None:
        self.driver.upload_json("config/run.json", config_payload)

    def materialize_checkpoint(self, key: str) -> str:
        if key.endswith("/"):
            target_dir = self.local_root("resume") / Path(key.rstrip("/")).name
            self.driver.download_prefix(key, target_dir)
            return str(target_dir)
        suffix = Path(key).suffix
        tmp = self._tempfile(prefix="resume-", suffix=suffix or ".ckpt")
        tmp_path = Path(tmp)
        self.driver.download_file(key, tmp_path)
        if suffix == ".json":
            payload = json.loads(tmp_path.read_text())
            metadata = payload.get("metadata", {})
            hf_dir = metadata.get("hf_checkpoint_dir")
            if hf_dir:
                target_dir = self.local_root("resume") / Path(hf_dir.rstrip("/")).name
                self.driver.download_prefix(hf_dir, target_dir)
                return str(target_dir)
        return tmp

    def sync_directory(self, directory: Path, remote_prefix: str) -> None:
        self.driver.sync_directory(directory, remote_prefix)

    def upload_file(self, source: Path, remote_key: str) -> None:
        self.driver.upload_file(remote_key, source)

    def create_tmp_file(self, prefix: str, suffix: str) -> str:
        return self._tempfile(prefix=prefix, suffix=suffix)

    def local_root(self, subdir: Optional[str] = None) -> Path:
        if subdir:
            path = self._local_root / subdir
            path.mkdir(parents=True, exist_ok=True)
            return path
        return self._local_root

    def _tempfile(self, prefix: str, suffix: str) -> str:
        fd, tmp = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=self._local_root)
        os.close(fd)
        return tmp


def main() -> None:
    logging.basicConfig(level=os.getenv("TRAINER_LOG_LEVEL", "INFO"))
    args, overrides = parse_args()
    manager = CheckpointManager(args)
    manager.ensure_layout()
    manager.persist_config({"framework": args.framework, "model": args.model_id, "overrides": overrides})
    runner = get_runner(args.framework)
    runner(args, overrides, manager, args.dataset_uri)


if __name__ == "__main__":
    main()
