import torch
import torch.nn as nn
import explainable_gnn as eg


class SemanticAttention(nn.Module):

    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


# SemanticAttention is use weighted beta to act as the attention weights
# Notice that beta is not related to the number of nodes in the graph
# So we can directly give the final beta to the inference model

class SemanticAttentionTranslation(eg.Module):
    def __init__(self, beta):
        super(SemanticAttentionTranslation, self).__init__()

        self.beta = nn.Parameter(beta)

    def forward(self, z):
        beta = self.beta.expand((z.shape[0],) + self.beta.shape)
        return (beta * z).sum(1)

