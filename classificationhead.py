import torch
from torch import nn
from labml_helpers.module import Module


class ClassificationHead(Module):
    """
    <a id="ClassificationHead"></a>

    ## MLP Classification Head

    This is the two layer MLP head to classify the image based on `[CLS]` token embedding.
    """
    def __init__(self, d_model: int, n_hidden: int, n_classes: int):
        """
        * `d_model` is the transformer embedding size
        * `n_hidden` is the size of the hidden layer
        * `n_classes` is the number of classes in the classification task
        """
        super().__init__()
        self.fc1 = nn.Sequential(
          # First layer
          nn.Linear(d_model, n_hidden),
          # Activation
          nn.ReLU(),
          # Second layer
          nn.Linear(n_hidden, n_classes),
        )
        self.fc2 = nn.Sequential(
          # First layer
          nn.Linear(d_model, n_hidden),
          # Activation
          nn.ReLU(),
          # Second layer
          nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        * `x` is the transformer encoding for `[CLS]` token
        """
        # First fully connected network
        x1 = self.fc1(x)
        # Second fully connected network
        x2 = self.fc2(x)

        #
        return x1,x2