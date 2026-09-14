import numpy as np
import time

class MultiHeadAttentionResearchModel:
    def __init__(self, embed_dim=128, num_heads=4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Initialize orthogonal weights
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.01

    def forward(self, X):
        start_time = time.time()
        
        # Linear projections
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # Scaled Dot-Product Attention
        scores = np.dot(Q, K.T) / np.sqrt(self.head_dim)
        attention_weights = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)), axis=-1, keepdims=True)
        context = np.dot(attention_weights, V)
        output = np.dot(context, self.W_o)
        
        latency = (time.time() - start_time) * 1000
        return output, attention_weights, latency

if __name__ == "__main__":
    model = MultiHeadAttentionResearchModel()
    sample_input = np.random.randn(1, 128)
    out, weights, lat = model.forward(sample_input)
    print(f"[*] Neural Attention Forward Pass Completed in {lat:.3f} ms")
    print(f"    Output Tensor Shape: {out.shape}")
