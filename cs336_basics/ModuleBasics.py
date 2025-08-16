import torch
import torch.nn as nn

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None): 
        # Call the superclass constructor
        super().__init__()
        
        # Store dimensions
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight parameter with correct shape (out_features, in_features)
        # This is W, not W^T, as requested in the comments
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        
        # Initialize using truncated normal distribution
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
        
        # Store as nn.Parameter so it's registered as a learnable parameter
        self.W = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear transformation
        # x shape: (..., in_features)
        # W shape: (out_features, in_features)
        # output shape: (..., out_features)
        
        # Using torch.einsum for the matrix multiplication
        # For batch processing: "...i,oi->...o" where:
        # ... represents any number of batch dimensions
        # i is in_features, o is out_features
        Y = torch.einsum('...i,oi->...o', x, self.W)
        
        return Y


        # To test your Linear module, implement the test adapter at [adapters.run_linear]. The adapter
        # should load the given weights into your Linear module. You can use Module.load_state_dict for
        # this purpose. Then, run uv run pytest -k test_linear.

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # Call the superclass constructor
        super().__init__()
        
        # Store dimensions
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create embedding weight matrix with shape (num_embeddings, embedding_dim)
        # This stores d_model (embedding_dim) as the final dimension as requested
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        
        # Initialize using truncated normal distribution
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
        
        # Store as nn.Parameter so it's registered as a learnable parameter
        self.embedding = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup the embedding vectors for the given token IDs
        # token_ids shape: (batch_size, sequence_length) - LongTensor of token indices
        # embedding shape: (num_embeddings, embedding_dim)
        # output shape: (batch_size, sequence_length, embedding_dim)
        
        # Use advanced indexing to lookup embeddings
        # self.embedding[token_ids] will broadcast correctly
        embeddings = self.embedding[token_ids]
        
        return embeddings

        # To test your implementation, implement the test adapter at [adapters.run_embedding]. Then, run
        # uv run pytest -k test_embedding.

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        # Construct the RMSNorm module. This function should accept the following parameters:
        # d_model: int Hidden dimension of the model
        # eps: float = 1e-5 Epsilon value for numerical stability
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        # Note: Remember to upcast your input to torch.float32 before performing the normalization (and
        # later downcast to the original dtype), as described above.
        # To test your implementation, implement the test adapter at [adapters.run_rmsnorm]. Then, run uv
        # run pytest -k test_rmsnorm.


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        # Construct the RoPE module and create buffers if needed.
        # theta: float Θ value for the RoPE
        # d_k: int dimension of query and key vectors
        # max_seq_len: int Maximum sequence length that will be inputted
        # device: torch.device | None = None Device to store the buffer on

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
        # Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        # Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        # assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        # positions of x along the sequence dimension.
        # You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        # along the sequence dimension.
        # To test your implementation, complete [adapters.run_rope] and make sure it passes uv run
        # pytest -k test_rope.