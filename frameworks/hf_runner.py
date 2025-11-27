"""Hugging Face Trainer implementation."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from .data_utils import download_dataset, split_corpus, stream_jsonl

logger = logging.getLogger(__name__)


def run(args, overrides: Dict[str, any], manager, dataset_uri: str) -> None:
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainerCallback, TrainingArguments, default_data_collator)

    class HFCheckpointUploader(TrainerCallback):
        def __init__(self, manager, output_dir: Path) -> None:
            self.manager = manager
            self.output_dir = output_dir
            self._synced: set[str] = set()

        def on_save(self, args, state, control, **kwargs):  # noqa: D401
            step = state.global_step
            if step is None:
                return
            ckpt_dir = self.output_dir / f"checkpoint-{step}"
            if not ckpt_dir.exists() or ckpt_dir.name in self._synced:
                return
            remote = f"checkpoints/{ckpt_dir.name}"
            self.manager.sync_directory(ckpt_dir, remote)
            meta = {"hf_checkpoint_dir": remote}
            self.manager.save_checkpoint(step, meta)
            self._synced.add(ckpt_dir.name)

    dataset_path = download_dataset(dataset_uri, args.run_id)
    tokenizer_name = overrides.get("tokenizer_name") or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_field = overrides.get("dataset_text_field", "text")
    train_lines, eval_lines = split_corpus(stream_jsonl(dataset_path, text_field=text_field))

    def tokenize(batch):
        return tokenizer(batch[text_field], truncation=True, max_length=int(overrides.get("max_seq_len", 2048)))

    train_dataset = Dataset.from_dict({text_field: train_lines}).map(tokenize, batched=True)
    eval_dataset = Dataset.from_dict({text_field: eval_lines}).map(tokenize, batched=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    output_dir = Path(os.getcwd()) / f"outputs-hf-{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=float(overrides.get("num_epochs", 1)),
        per_device_train_batch_size=int(overrides.get("batch_size", 2)),
        per_device_eval_batch_size=int(overrides.get("eval_batch_size", overrides.get("batch_size", 2))),
        gradient_accumulation_steps=int(overrides.get("gradient_accumulation", 1)),
        learning_rate=float(overrides.get("lr", 5e-5)),
        weight_decay=float(overrides.get("weight_decay", 0.0)),
        warmup_steps=int(overrides.get("warmup_steps", 0)),
        logging_steps=int(overrides.get("logging_steps", 10)),
        save_steps=int(overrides.get("save_steps", 100)),
        evaluation_strategy="steps",
        eval_steps=int(overrides.get("eval_steps", 100)),
        bf16=overrides.get("precision", "bf16").lower() == "bf16",
        fp16=overrides.get("precision", "bf16").lower() == "fp16",
        report_to=[],
    )

    resume_checkpoint: Optional[str] = None
    latest = manager.latest_checkpoint()
    if args.resume and latest:
        resume_checkpoint = manager.materialize_checkpoint(latest)
        logger.info("Resuming HF Trainer from %s", resume_checkpoint)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        callbacks=[HFCheckpointUploader(manager, output_dir)],
    )

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    trainer.save_model()
    manager.sync_directory(output_dir, "checkpoints/full-run")
