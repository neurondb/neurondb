"""
Training script for Olympus-MAC language modeling (WikiText-103).
Usage: from project root: python experiments/lm/train_lm.py --config experiments/lm/config_small.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from experiments.lm.data_lm import get_lm_dataloader, get_tokenizer, load_wikitext103
from olympus import OlympusMAC


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/lm/config_small.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device)
    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    if vocab_size != cfg["model"]["vocab_size"]:
        cfg["model"]["vocab_size"] = vocab_size

    print("Loading WikiText-103...")
    token_ids = load_wikitext103(tokenizer, max_length=cfg["data"]["max_length"], split="train")
    block_size = min(cfg["data"]["max_length"], len(token_ids) // 2)
    if block_size < 32:
        block_size = 32
    loader = get_lm_dataloader(
        token_ids,
        block_size=block_size,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
    )

    model = OlympusMAC(**cfg["model"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    accum = cfg["training"].get("grad_accum_steps", 1)
    log_interval = cfg["training"].get("log_interval", 10)
    eval_interval = cfg["training"].get("eval_interval", 100)
    save_interval = cfg["training"].get("save_interval", 500)

    global_step = 0
    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        running_loss = 0.0
        opt.zero_grad()
        for bi, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            logits, loss, info = model(x, labels=y, return_memory_loss=True)
            if loss is None:
                continue
            (loss / accum).backward()
            running_loss += loss.item()
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                global_step += 1
                if global_step % log_interval == 0:
                    avg = running_loss / (log_interval * accum)
                    writer.add_scalar("train/loss", avg, global_step)
                    if info:
                        writer.add_scalar("train/mem_loss", info.get("mem_loss", 0), global_step)
                    running_loss = 0.0
                    pbar.set_postfix(loss=f"{avg:.4f}")
                if global_step % save_interval == 0 and global_step > 0:
                    ckpt = out_dir / f"ckpt_{global_step}.pt"
                    torch.save({"model": model.state_dict(), "step": global_step}, ckpt)
        if (epoch + 1) % max(1, eval_interval // len(loader)) == 0:
            torch.save(model.state_dict(), out_dir / "last.pt")
    writer.close()
    print(f"Done. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
