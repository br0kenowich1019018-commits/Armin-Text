import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------
# Device
# ---------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x, hidden=None):
        x = self.embedding(x)

        output, hidden = self.gru(
            x,
            hidden
        )

        output = self.dropout(output)

        logits = self.fc(output)

        return logits, hidden


# ---------------------------------
# Load model
# ---------------------------------

checkpoint = torch.load(
    "ArminText-v1.pt",
    map_location=device,
    weights_only=True
)

word2id = checkpoint["word2id"]
id2word = checkpoint["id2word"]
config = checkpoint["config"]

PAD_ID = word2id["<PAD>"]
UNK_ID = word2id["<UNK>"]
BOS_ID = word2id["<BOS>"]
EOS_ID = word2id["<EOS>"]

SEQ_LEN = config["seq_len"]

model = ArminText(
    vocab_size=config["vocab_size"],
    embedding_dim=config["embedding_dim"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    dropout=0.2
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ---------------------------------
# Text generation
# ---------------------------------

def generate_text(
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=20
):
    words = prompt.split()

    input_ids = [
        word2id.get(word, UNK_ID)
        for word in words
    ]

    if not input_ids:
        input_ids = [BOS_ID]

    generated = input_ids.copy()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            context = generated[-SEQ_LEN:]

            x = torch.tensor(
                [context],
                dtype=torch.long,
                device=device
            )

            logits, _ = model(x)

            next_logits = logits[0, -1]

            next_logits = next_logits / temperature

            values, indices = torch.topk(
                next_logits,
                k=min(top_k, len(next_logits))
            )

            probs = F.softmax(
                values,
                dim=-1
            )

            selected = torch.multinomial(
                probs,
                num_samples=1
            )

            next_token = indices[selected].item()

            if next_token == EOS_ID:
                break

            generated.append(next_token)

    result = [
        id2word.get(token, "<UNK>")
        for token in generated
        if token not in [BOS_ID, EOS_ID, PAD_ID]
    ]

    return " ".join(result)


# ---------------------------------
# Command-line interface
# ---------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Persian text with ArminText-v1"
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="Persian prompt"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20
    )

    args = parser.parse_args()

    result = generate_text(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    print("\nPrompt:")
    print(args.prompt)

    print("\nArminText:")
    print(result)


if __name__ == "__main__":
    main()
