import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.utils import clone_module_list
from labml_nn.transformers import TransformerLayer

from embeddings import PatchEmbeddings, LearnedPositionalEmbeddings
from classificationhead import ClassificationHead
from transformer import VisionTransformer

class DWC():
  """
  This creates the DWC model branch
  """
  def __init__(self, d_model=768, patch_size=16, in_channels=3, max_len = 2500,
              n_hidden=512, n_classes=7800, transformer_layer=TransformerLayer,
              n_layers=12, patch_emb=PatchEmbeddings,
              pos_emb=LearnedPositionalEmbeddings, classification=ClassificationHead):
      
      print(VisionTransformer(TransformerLayer=transformer_layer,n_layers= n_layers,
                                    patch_emb= patch_emb, pos_emb=pos_emb, classification= classification))
