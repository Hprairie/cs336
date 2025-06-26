import torch
import torch.nn as nn
import einx


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        self.dtype = dtype
        self.device = device
        self._init_weights(0, float(2 / (in_features + out_features)))
    
    def _init_weights(self, mean, var) -> None:
        nn.init.trunc_normal_(self.weight, mean, var, -3 * var, 3 * var)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return einx.dot("... in_features, out_features in_features -> ... out_features", x, self.weight)

class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int, 
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None: 
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        self._init_embeddings(0, 1)
    
    def _init_embeddings(self, mean, var):
        nn.init.trunc_normal_(self.embeddings, mean, var, -3 * var, 3 * var)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return einx.get_at("[v] d, ... t -> ... t d", self.embeddings, token_ids)

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None: 
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(d_model, dtype=dtype, device=device)
        )
        self._init_weights()
        self.eps = eps
    
    def _init_weights(self) -> None:
        nn.init.ones_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        def _rms_op(input):
            return torch.rsqrt(torch.mean(input ** 2) + self.eps) * input * self.weight
        return einx.vmap("... [a] -> ... [a] ", x, op=_rms_op)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None, 
    ) -> None:
        super().__init__()
        self.weight1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.weight2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.weight3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.weight2(self.weight3(x) * silu(self.weight1(x)))


if __name__ == "__main__":
    # x = torch.ones(128)
    # layer = Linear(64, 128)
    # print(layer(x))
    # x = torch.arange(128)
    # layer = Embedding(256, 12)
    # print(layer(x))
    # x = torch.randn(128)
    # layer = RMSNorm(128)
    # print(layer(x))
    x = torch.randn(128)
    layer =SwiGLU(128, 256)
    print(layer(x))