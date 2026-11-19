
import torch
import torch.nn as nn

from config import (
    block_size,
    dropout,
    grad_clip,
    device,
    eval_interval,
    eval_iters,
    learning_rate,
    max_iters,
)

from model import TransformerLanguageModel
from data import get_batch, vocab_size


SEED = 42
torch.manual_seed(SEED)


model = TransformerLanguageModel(
    vocab_size, 384, 6, 3, block_size, dropout
).to(device)
print("Model device:", next(model.parameters()).device)

optim = torch.optim.AdamW(
    model.parameters(), lr=learning_rate
)
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}

    for split in ["train", "val"]:
        total_loss = 0.0

        for _ in range(eval_iters):
            x, targets = get_batch(split)

            logits = model(x)
            logits = logits.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(logits, targets)
            total_loss += loss.item()

        losses[split] = total_loss / eval_iters

    model.train()
    return losses


# Training loop
for iter in range(max_iters):

    x, targets = get_batch("train")

    logits = model(x)
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)

    loss = criterion(logits, targets)

    optim.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip
    )

    optim.step()

    
    if iter % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {iter}: "
            f"Train Loss: {losses['train']:.4f}, "
            f"Validation Loss: {losses['val']:.4f}"
        )



torch.save(model.state_dict(), "transformer.pth")
