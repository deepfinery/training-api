"""Meta/custom PyTorch trainer leveraging FSDP when multiple GPUs are present."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

from .data_utils import download_dataset, split_corpus, stream_jsonl

logger = logging.getLogger(__name__)


def run(args, overrides: Dict[str, any], manager, dataset_uri: str) -> None:
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class TextDataset(Dataset):
        def __init__(self, lines, tokenizer, max_length: int) -> None:
            texts = list(lines)
            self.examples = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

        def __len__(self) -> int:  # noqa: D401
            return self.examples.input_ids.size(0)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            return {
                "input_ids": self.examples.input_ids[idx],
                "labels": self.examples.input_ids[idx].clone(),
            }

    def _init_process_group() -> None:
        if dist.is_initialized():
            return
        dist.init_process_group(backend="nccl")

    dataset_path = download_dataset(dataset_uri, args.run_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = int(overrides.get("max_seq_len", 2048))
    train_lines, eval_lines = split_corpus(stream_jsonl(dataset_path))

    train_dataset = TextDataset(train_lines, tokenizer, max_len)
    eval_dataset = TextDataset(eval_lines, tokenizer, max_len)

    per_device_batch = int(overrides.get("batch_size", 1))
    epochs = int(overrides.get("num_epochs", 1))
    lr = float(overrides.get("lr", 2e-5))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    _init_process_group()

    model = AutoModelForCausalLM.from_pretrained(args.model_id).cuda()

    if dist.get_world_size() > 1:
        auto_wrap_policy = transformer_auto_wrap_policy(AutoModelForCausalLM)
        model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
    else:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    train_loader = DataLoader(train_dataset, batch_size=per_device_batch, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=per_device_batch)

    manager.ensure_layout()
    start_step = 0
    latest = manager.latest_checkpoint()
    if args.resume and latest:
        state_path = manager.materialize_checkpoint(latest)
        logger.info("Loading Meta checkpoint from %s", state_path)
        checkpoint = torch.load(state_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint.get("step", 0)

    global_step = start_step
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            if global_step % int(overrides.get("save_steps", 100)) == 0:
                _save_meta_checkpoint(model, optimizer, global_step, manager)

        evaluate(model, eval_loader, loss_fn)

    _save_meta_checkpoint(model, optimizer, global_step, manager)


def _save_meta_checkpoint(model, optimizer, step: int, manager) -> None:
    import torch

    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    tmp = Path(manager.create_tmp_file(prefix=f"meta-step-{step:06d}", suffix=".pt"))
    torch.save(state, tmp)
    manager.upload_file(tmp, f"checkpoints/step-{step:06d}.pt")


def evaluate(model, loader, loss_fn) -> None:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model(inputs, labels=labels)
            losses.append(outputs.loss.item())
    if losses:
        logger = logging.getLogger(__name__)
        logger.info("Eval loss %.4f", sum(losses) / len(losses))
