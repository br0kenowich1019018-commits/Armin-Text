import argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm


# ---------------------------------
# Device
# ---------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------
# Arguments
# ---------------------------------

parser = argparse.ArgumentParser(
    description="Train ArminText Persian language model"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=5
)

parser.add_argument(
    "--max_stories",
    type=int,
    default=8000
)

parser.add_argument(
    "--max_tokens",
    type=int,
    default=600_000
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=64
)

args = parser.parse_args()


# ---------------------------------
# Load dataset
# ---------------------------------

print("\nLoading dataset...")

dataset = load_dataset(
    "taesiri/TinyStories-Farsi",
    split="train",
    streaming=True
)

sample = next(iter(dataset))


def contains_persian(text):
    if not isinstance(text, str):
        return False

    return any(
        "\u0600" <= ch <= "\u06FF"
        for ch in text
    )


persian_column = None

for key, value in sample.items():
    if contains_persian(value):
        persian_column = key
        break


if persian_column is None:
    raise ValueError("Persian text column was not found.")


print("Persian column:", persian_column)


# ---------------------------------
# Collect training texts
# ---------------------------------

texts = []
total_tokens = 0

dataset = load_dataset(
    "taesiri/TinyStories-Farsi",
    split="train",
    streaming=True
)

progress = tqdm(
    dataset,
    total=args.max_stories,
    desc="Collecting stories"
)

for row in progress:

    text = row[persian_column]

    if not isinstance(text, str):
        continue

    text = text.strip()

    if len(text) < 50:
        continue

    words = text.split()

    if len(words) < 10:
        continue

    if total_tokens + len(words) > args.max_tokens:
        break

    texts.append(
        " ".join(words)
    )

    total_tokens += len(words)

    if len(texts) >= args.max_stories:
        break


print("\nStories:", len(texts))
print("Approx. words:", total_tokens)


# ---------------------------------
# Train / validation split
# ---------------------------------

split_index = int(
    len(texts) * 0.9
)

train_texts = texts[:split_index]
val_texts = texts[split_index:]

print("Train stories:", len(train_texts))
print("Validation stories:", len(val_texts))


# ---------------------------------
# Build vocabulary
# ---------------------------------

counter = Counter()

for text in train_texts:
    counter.update(text.split())


MAX_VOCAB = 12_000
MIN_FREQ = 2

special_tokens = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]

word2id = {
    token: i
    for i, token in enumerate(special_tokens)
}

for word, freq in counter.most_common():

    if freq < MIN_FREQ:
        break

    if len(word2id) >= MAX_VOCAB:
        break

    word2id[word] = len(word2id)


id2word = {
    idx: word
    for word, idx in word2id.items()
}

vocab_size = len(word2id)

PAD_ID = word2id["<PAD>"]
UNK_ID = word2id["<UNK>"]
BOS_ID = word2id["<BOS>"]
EOS_ID = word2id["<EOS>"]

print("Vocabulary size:", vocab_size)


# ---------------------------------
# Tokenization
# ---------------------------------

def encode_text(text):

    words = text.split()

    ids = [BOS_ID]

    for word in words:
        ids.append(
            word2id.get(word, UNK_ID)
        )

    ids.append(EOS_ID)

    return ids


train_ids = []

for text in train_texts:
    train_ids.extend(
        encode_text(text)
    )


val_ids = []

for text in val_texts:
    val_ids.extend(
        encode_text(text)
    )


print("Train tokens:", len(train_ids))
print("Validation tokens:", len(val_ids))


# ---------------------------------
# Dataset
# ---------------------------------

SEQ_LEN = 32


class NextWordDataset(Dataset):

    def __init__(self, token_ids, seq_len):
        self.data = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):

        x = self.data[
            index:index + self.seq_len
        ]

        y = self.data[
            index + 1:index + self.seq_len + 1
        ]

        return x, y


train_dataset = NextWordDataset(
    train_ids,
    SEQ_LEN
)

val_dataset = NextWordDataset(
    val_ids,
    SEQ_LEN
)


train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ---------------------------------
# Model
# ---------------------------------

class ArminText(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=192,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        output, hidden = self.gru(x)

        output = self.dropout(output)

        logits = self.fc(output)

        return logits, hidden


model = ArminText(
    vocab_size=vocab_size
).to(device)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nParameters: {total_params:,}"
)


# ---------------------------------
# Training setup
# ---------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.002,
    weight_decay=0.01
)


# ---------------------------------
# Training loop
# ---------------------------------

for epoch in range(args.epochs):

    model.train()

    total_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{args.epochs}"
    )

    for x, y in progress:

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits, _ = model(x)

        loss = criterion(
            logits.reshape(
                -1,
                vocab_size
            ),
            y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    avg_train_loss = (
        total_loss / len(train_loader)
    )


    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            logits, _ = model(x)

            loss = criterion(
                logits.reshape(
                    -1,
                    vocab_size
                ),
                y.reshape(-1)
            )

            val_loss += loss.item()


    avg_val_loss = (
        val_loss / len(val_loader)
    )


    print(
        f"\nEpoch {epoch + 1}: "
        f"Train Loss = {avg_train_loss:.4f} | "
        f"Val Loss = {avg_val_loss:.4f}"
    )


# ---------------------------------
# Save checkpoint
# ---------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "word2id": word2id,

        "id2word": id2word,

        "config": {
            "vocab_size": vocab_size,
            "embedding_dim": 192,
            "hidden_dim": 256,
            "num_layers": 2,
            "seq_len": SEQ_LEN
        }
    },
    "ArminText-v1.pt"
)


print(
    "\n✅ Training finished!"
)

print(
    "✅ Saved model: ArminText-v1.pt"
)
