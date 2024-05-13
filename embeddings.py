#This is a [PyTorch](https://pytorch.org) implementation of the paper
#[An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/abs/2010.11929).
# Modified from LabML implementation
import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers import TransformerLayer
from labml_nn.utils import clone_module_list

class PatchEmbeddings(Module):
    """
    <a id="PatchEmbeddings"></a>

    ## Get patch embeddings

    The paper splits the image into patches of equal size and do a linear transformation
    on the flattened pixels for each patch.

    We implement the same thing through a convolution layer, because it's simpler to implement.
    """

    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        """
        * `d_model` is the transformer embeddings size
        * `patch_size` is the size of the patch
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()

        # We create a convolution layer with a kernel size and and stride length equal to patch size.
        # This is equivalent to splitting the image into patches and doing a linear
        # transformation on each patch.
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Apply convolution layer
        x = self.conv(x)
        # Get the shape.
        bs, c, h, w = x.shape
        # Rearrange to shape `[patches, batch_size, d_model]`
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c)

        # Return the patch embeddings
        return x


class LearnedPositionalEmbeddings(Module):
    """
    <a id="LearnedPositionalEmbeddings"></a>

    ## Add parameterized positional encodings

    This adds learned positional embeddings to patch embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5_000):
        """
        * `d_model` is the transformer embeddings size
        * `max_len` is the maximum number of patches
        """
        super().__init__()
        # Positional embeddings for each location
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the patch embeddings of shape `[patches, batch_size, d_model]`
        """
        # Get the positional embeddings for the given patches
        pe = self.positional_encodings[:x.shape[0]]
        # Add to patch embeddings and return
        return x + pe
