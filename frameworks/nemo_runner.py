"""Nemo-style trainer implemented with PyTorch Lightning."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

from .data_utils import download_dataset, split_corpus, stream_jsonl

logger = logging.getLogger(__name__)


def run(args, overrides: Dict[str, any], manager, dataset_uri: str) -> None:
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:  # optional integration
        from nemo.utils import exp_manager
    except Exception:  # noqa: BLE001
        exp_manager = None  # type: ignore

    class NemoDataset(Dataset):
        def __init__(self, lines, tokenizer, max_length: int) -> None:
            tokenized = tokenizer(
                lines,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.ids = tokenized.input_ids

        def __len__(self) -> int:  # noqa: D401
            return self.ids.size(0)

        def __getitem__(self, idx: int):
            tensor = self.ids[idx]
            return {"input_ids": tensor, "labels": tensor.clone()}

    class NemoLightningModule(pl.LightningModule):
        def __init__(self, model_name: str, lr: float) -> None:
            super().__init__()
            self.save_hyperparameters()
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.lr = lr

        def training_step(self, batch, batch_idx):  # noqa: D401
            outputs = self.model(**batch)
            loss = outputs.loss
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):  # noqa: D401
            outputs = self.model(**batch)
            self.log("val_loss", outputs.loss, prog_bar=True)

        def configure_optimizers(self):  # noqa: D401
            return torch.optim.AdamW(self.parameters(), lr=self.lr)

    dataset_path = download_dataset(dataset_uri, args.run_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(overrides.get("max_seq_len", 2048))
    train_lines, eval_lines = split_corpus(stream_jsonl(dataset_path))
    train_dataset = NemoDataset(train_lines, tokenizer, max_length)
    eval_dataset = NemoDataset(eval_lines, tokenizer, max_length)

    per_device_batch = int(overrides.get("batch_size", 1))
    num_workers = int(overrides.get("dataloader_workers", 2))

    train_loader = DataLoader(train_dataset, batch_size=per_device_batch, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=per_device_batch, num_workers=num_workers)

    module = NemoLightningModule(args.model_id, float(overrides.get("lr", 2e-5)))
    strategy = "ddp" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "auto"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=int(os.environ.get("GPUS_PER_NODE", 1)),
        strategy=strategy,
        max_epochs=int(overrides.get("num_epochs", 1)),
        log_every_n_steps=int(overrides.get("log_every_n_steps", 10)),
        enable_checkpointing=True,
        default_root_dir=f"nemo-outputs-{args.run_id}",
    )

    if exp_manager:
        cfg = {"exp_dir": manager.local_root("nemo"), "name": args.run_id}
        exp_manager(trainer, cfg)  # type: ignore[arg-type]

    ckpt_path = None
    latest = manager.latest_checkpoint()
    if args.resume and latest:
        ckpt_path = manager.materialize_checkpoint(latest)
        logger.info("Resuming NeMo Lightning module from %s", ckpt_path)

    trainer.fit(module, train_loader, eval_loader, ckpt_path=ckpt_path)
    output_dir = Path(trainer.default_root_dir)
    manager.sync_directory(output_dir, "checkpoints")
