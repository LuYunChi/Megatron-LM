import torch
import torch.nn as nn
from torch.optim import Adam


class TinyGPT(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=64, num_layers=2, num_heads=4, max_seq_len=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_seq_len, hidden_size)

        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        pos_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)

        x = self.embed_tokens(tokens) + self.embed_positions(pos_ids)
        x = self.transformer(x)  # [B, T, H]
        logits = self.lm_head(x)  # [B, T, vocab]
        return logits


def get_dummy_batch(batch_size, seq_len, vocab_size, device):
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = tokens.clone()
    return tokens, labels


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size, seq_len, batch_size = 128, 64, 8
    model = TinyGPT(vocab_size, hidden_size=64, num_layers=2, num_heads=4, max_seq_len=seq_len).to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for step in range(3):
        tokens, labels = get_dummy_batch(batch_size, seq_len, vocab_size, device)

        optim.zero_grad()
        logits = model(tokens)  # [B, T, vocab]

        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optim.step()

        print(f"[Step {step}] Loss: {loss.item():.4f}")