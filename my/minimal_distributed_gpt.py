import torch
from torch.optim import Adam

# Megatron-Core imports
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec


# ---------------------------
# Initialize trivial parallel state (TP=1, PP=1)
# ---------------------------
def init_single_process():
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)


# ---------------------------
# Build a tiny GPT model
# ---------------------------
def build_model(vocab_size=128, seq_len=64):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
    )


# ---------------------------
# Dummy toy data
# ---------------------------
def get_dummy_batch(batch_size, seq_len, vocab_size, device):
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = tokens.clone()
    return tokens, position_ids, attention_mask, labels


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_single_process()

    vocab_size, seq_len, batch_size = 128, 64, 8
    model = build_model(vocab_size, seq_len).to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    for step in range(3):
        tokens, pos, mask, labels = get_dummy_batch(batch_size, seq_len, vocab_size, device)

        optim.zero_grad()
        # Megatron GPTModel forward returns per-token loss values when labels are passed
        output = model(tokens, pos, mask, labels=labels)
        loss = output.mean()
        loss.backward()
        optim.step()

        print(f"[Step {step}] Loss: {loss.item():.4f}")