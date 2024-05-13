import torch
from torch import nn
from labml_helpers.module import Module
from labml_nn.transformers import TransformerLayer
from embeddings import PatchEmbeddings, LearnedPositionalEmbeddings
from classificationhead import ClassificationHead

class VisionTransformer(Module):
    """
    ## Vision Transformer

    This combines the [patch embeddings](#PatchEmbeddings),
    [positional embeddings](#LearnedPositionalEmbeddings),
    transformer and the [classification head](#ClassificationHead).
    """
    def __init__(self, transformer_layer: TransformerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings, pos_emb: LearnedPositionalEmbeddings,
                 classification: ClassificationHead):
        """
        * `transformer_layer` is a copy of a single [transformer layer](../models.html#TransformerLayer).
         We make copies of it to make the transformer with `n_layers`.
        * `n_layers` is the number of [transformer layers](../models.html#TransformerLayer).
        * `patch_emb` is the [patch embeddings layer](#PatchEmbeddings).
        * `pos_emb` is the [positional embeddings layer](#LearnedPositionalEmbeddings).
        * `classification` is the [classification head](#ClassificationHead).
        """
        super().__init__()
        # Patch embeddings
        self.patch_emb = patch_emb
        self.pos_emb = pos_emb
        # Classification head
        self.classification = classification
        # Make copies of the transformer layer
        self.transformer_layers = clone_module_list(transformer_layer, n_layers)

        # `[CLS]` token embedding
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, transformer_layer.size), requires_grad=True)
        # Final normalization layer
        self.ln = nn.LayerNorm([transformer_layer.size])

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Get patch embeddings. This gives a tensor of shape `[patches, batch_size, d_model]`
        x = self.patch_emb(x)
        # Concatenate the `[CLS]` token embeddings before feeding the transformer
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])
        # Add positional embeddings
        x = self.pos_emb(x)

        # Pass through transformer layers with no attention masking
        for layer in self.transformer_layers:
            x = layer(x=x, mask=None)

        # Get the transformer output of the `[CLS]` token (which is the first in the sequence).
        x = x[0]

        # Layer normalization
        x = self.ln(x)

        # Classification head, to get logits
        x = self.classification(x)

        #
        return x