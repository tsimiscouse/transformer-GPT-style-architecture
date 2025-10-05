import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

def softmax(x, axis=-1):
    # Numerical stability by subtracting max
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    # Layer normalization: (x - mean) / std * gamma + beta
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def gelu(x):
    # Approximation of GELU activation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def create_causal_mask(seq_len):
    # Upper triangular matrix with -inf (blocks future tokens)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    return mask


class TokenEmbedding:
    """Converts token indices to dense vector representations"""

    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Xavier initialization
        self.weight = np.random.randn(vocab_size, d_model) * np.sqrt(2.0 / (vocab_size + d_model))

    def forward(self, x):
        # Lookup embedding for each token
        return self.weight[x]


class PositionalEncoding:
    """Sinusoidal positional encoding"""

    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model

        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe

    def forward(self, x):
        # Add positional encoding to embeddings
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]


class RotaryPositionalEmbedding:
    """RoPE - Rotary Position Embedding (BONUS)"""

    def __init__(self, d_model, max_len=5000):
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.d_model = d_model

        # Compute rotation frequencies
        inv_freq = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))
        position = np.arange(max_len)
        freqs = np.outer(position, inv_freq)

        # Cache cosine and sine values
        self.cos_cached = np.cos(freqs)
        self.sin_cached = np.sin(freqs)

    def rotate_half(self, x):
        # Rotate by swapping pairs and negating
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        result = np.zeros_like(x)
        result[..., ::2] = -x2
        result[..., 1::2] = x1
        return result

    def forward(self, x):
        # Apply rotary position embedding
        seq_len = x.shape[1]
        cos = np.repeat(self.cos_cached[:seq_len, :], 2, axis=-1)
        sin = np.repeat(self.sin_cached[:seq_len, :], 2, axis=-1)
        return x * cos + self.rotate_half(x) * sin


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled Dot-Product Attention mechanism"""
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Apply mask if provided (for causal attention)
    if mask is not None:
        scores = scores + mask

    # Get attention weights via softmax
    attention_weights = softmax(scores, axis=-1)

    # Apply attention to values
    output = np.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention:
    """Multi-Head Attention with Q, K, V projections"""

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize projection weights
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

    def split_heads(self, x):
        # Reshape to separate heads: [batch, seq_len, num_heads, d_k]
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to [batch, num_heads, seq_len, d_k]
        return x.transpose(0, 2, 1, 3)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Linear projections and split into heads
        Q = self.split_heads(np.matmul(x, self.W_q))
        K = self.split_heads(np.matmul(x, self.W_k))
        V = self.split_heads(np.matmul(x, self.W_v))

        # Reshape for batch processing
        Q = Q.reshape(batch_size * self.num_heads, seq_len, self.d_k)
        K = K.reshape(batch_size * self.num_heads, seq_len, self.d_k)
        V = V.reshape(batch_size * self.num_heads, seq_len, self.d_k)

        # Apply attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Reshape back
        attn_output = attn_output.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        attn_weights = attn_weights.reshape(batch_size, self.num_heads, seq_len, seq_len)

        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Final projection
        output = np.matmul(attn_output, self.W_o)

        return output, attn_weights


class FeedForwardNetwork:
    """Two-layer MLP with GELU activation"""

    def __init__(self, d_model, d_ff):
        # Initialize two linear layers
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # First layer with GELU activation
        hidden = gelu(np.matmul(x, self.W1) + self.b1)
        # Second layer
        output = np.matmul(hidden, self.W2) + self.b2
        return output


class DecoderBlock:
    """Decoder block with Pre-Layer Normalization"""

    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        # Layer normalization parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def forward(self, x, mask=None):
        # Pre-norm Multi-Head Attention + Residual
        norm_x = layer_norm(x, self.gamma1, self.beta1)
        attn_output, attn_weights = self.attention.forward(norm_x, mask)
        x = x + attn_output

        # Pre-norm FFN + Residual
        norm_x = layer_norm(x, self.gamma2, self.beta2)
        ffn_output = self.ffn.forward(norm_x)
        x = x + ffn_output

        return x, attn_weights


class GPTTransformer:
    """Complete GPT-style Transformer model"""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_len=512, use_rope=False, weight_tying=False):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_rope = use_rope
        self.weight_tying = weight_tying

        # Token embedding
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        # Positional encoding (choose between standard or RoPE)
        if use_rope:
            self.positional_encoding = RotaryPositionalEmbedding(d_model, max_len)
        else:
            self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Stack of decoder blocks
        self.decoder_blocks = [
            DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ]

        # Final layer norm
        self.gamma_final = np.ones(d_model)
        self.beta_final = np.zeros(d_model)

        # Output projection (with optional weight tying)
        if weight_tying:
            self.output_projection = self.token_embedding.weight.T
        else:
            self.output_projection = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)

    def forward(self, x):
        # x: token indices [batch, seq_len]
        # Returns: logits [batch, seq_len, vocab_size],
        #          next_token_probs [batch, vocab_size],
        #          attention_weights (list of attention matrices)

        batch_size, seq_len = x.shape

        # Embedding + positional encoding
        embeddings = self.token_embedding.forward(x)
        x = self.positional_encoding.forward(embeddings)

        # Create causal mask
        mask = create_causal_mask(seq_len)

        # Pass through decoder blocks
        all_attention_weights = []
        for decoder_block in self.decoder_blocks:
            x, attn_weights = decoder_block.forward(x, mask)
            all_attention_weights.append(attn_weights)

        # Final layer norm
        x = layer_norm(x, self.gamma_final, self.beta_final)

        # Project to vocabulary size
        logits = np.matmul(x, self.output_projection)

        # Get next token probabilities (softmax of last position)
        last_token_logits = logits[:, -1, :]
        next_token_probs = softmax(last_token_logits, axis=-1)

        return logits, next_token_probs, all_attention_weights


def visualize_attention(attention_weights, layer_idx=0, head_idx=0):
    """Visualize single attention head as heatmap"""
    attn = attention_weights[layer_idx][0, head_idx, :, :]

    plt.figure(figsize=(8, 6))
    sns.heatmap(attn, cmap='viridis', annot=True, fmt='.2f', cbar=True)
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.show()


def plot_attention_statistics(attention_weights):
    """Plot average attention across all heads for each layer"""
    num_layers = len(attention_weights)

    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))
    if num_layers == 1:
        axes = [axes]

    for layer_idx, attn in enumerate(attention_weights):
        avg_attn = np.mean(attn, axis=(0, 1))

        sns.heatmap(avg_attn, cmap='Blues', cbar=True, ax=axes[layer_idx])
        axes[layer_idx].set_title(f'Layer {layer_idx}\nAvg Attention')
        axes[layer_idx].set_xlabel('Key')
        axes[layer_idx].set_ylabel('Query')

    plt.tight_layout()
    plt.show()


def test_transformer():
    """Test function to verify all components"""

    # Configuration
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    seq_len = 10
    batch_size = 2

    print("="*70)
    print("TRANSFORMER TESTING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Model Dimension: {d_model}")
    print(f"  Number of Heads: {num_heads}")
    print(f"  Number of Layers: {num_layers}")
    print(f"  FFN Dimension: {d_ff}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Batch Size: {batch_size}")

    # Create models
    print("\n" + "="*70)
    print("=== Testing Standard Transformer ===")
    model_standard = GPTTransformer(vocab_size, d_model, num_heads, num_layers, d_ff,
                                   use_rope=False, weight_tying=False)

    print("\n=== Testing RoPE Transformer ===")
    model_rope = GPTTransformer(vocab_size, d_model, num_heads, num_layers, d_ff,
                               use_rope=True, weight_tying=False)

    print("\n=== Testing Weight-Tied Transformer ===")
    model_tied = GPTTransformer(vocab_size, d_model, num_heads, num_layers, d_ff,
                               use_rope=False, weight_tying=True)

    # Input tokens (simple tokenization as list of numbers)
    tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\nInput tokens:\n{tokens}")

    # Test each model
    for model_name, model in [("Standard", model_standard),
                               ("RoPE", model_rope),
                               ("Weight-Tied", model_tied)]:
        print("\n" + "="*70)
        print(f"--- {model_name} Model ---")

        logits, next_token_probs, attention_weights = model.forward(tokens)

        print(f"Input tokens shape: {tokens.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Next token probabilities shape: {next_token_probs.shape}")
        print(f"Number of attention weight matrices: {len(attention_weights)}")
        print(f"Attention weights shape per layer: {attention_weights[0].shape}")
        print(f"Sum of next token probabilities: {np.sum(next_token_probs, axis=-1)}")
        print(f"  (should be ~1.0 for each sample)")

        # Top-5 predicted tokens
        for b in range(batch_size):
            top_5 = np.argsort(next_token_probs[b])[-5:][::-1]
            print(f"\n  Sample {b} - Top 5 predicted tokens:")
            for i, token_id in enumerate(top_5, 1):
                print(f"    {i}. Token {token_id}: {next_token_probs[b, token_id]:.4f}")

        if model_name == "Weight-Tied":
            print(f"\nWeight tying check:")
            print(f"  Embedding weight shape: {model.token_embedding.weight.shape}")
            print(f"  Output projection shape: {model.output_projection.shape}")
            print(f"  Weight tying enabled: {model.weight_tying}")

    # Visualization demo
    print("\n" + "="*70)
    print("=== Attention Visualization Demo ===")
    print("\nGenerating attention heatmap...")
    visualize_attention(attention_weights, layer_idx=0, head_idx=0)

    print("\nGenerating attention statistics...")
    plot_attention_statistics(attention_weights)

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)

    return True

if __name__ == "__main__":
    test_transformer()
