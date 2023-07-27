import torch
from einops import rearrange, reduce


class Vision:
    def __init__(self):
        self.weight = torch.randn((1024, 3 * 16 * 16))
        self.weight.requires_grad_(True)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = rearrange(image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=16, pw=16)
        # convolution
        representation = torch.einsum("b n p, d p -> b n d", image, self.weight) 
        # pooling
        representation = reduce(representation, "b n d -> b d", "mean")
        return representation