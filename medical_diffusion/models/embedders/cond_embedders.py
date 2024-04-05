import torch.nn as nn


class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_classes, emb_dim)

    def forward(self, condition):
        c = self.embedding(condition) #[B,] -> [B, C]
        return c



