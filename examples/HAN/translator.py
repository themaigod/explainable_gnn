# HAN has a simple structure, which is a stack of an attention layer and a GAT layer

# notice that the attention layer is for the number of meta paths

# after training, the attention parameters are stable for the given meta paths


import torch
import torch.nn as nn
import torch.nn.functional as F
import explainable_gnn as eg
import dgl.nn.pytorch


class SemanticAttentionTranslation(eg.Module):
    def __init__(self, beta):
        super(SemanticAttentionTranslation, self).__init__()

        self.beta = nn.Parameter(beta)

    def forward(self, z):
        beta = self.beta.expand((z.shape[0],) + self.beta.shape)
        return (beta * z).sum(1)

    def set_beta(self, beta):
        self.beta = nn.Parameter(beta)
        return self.beta


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout,
                 analyze=True
                 ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(
                dgl.nn.pytorch.GATConv(in_size, out_size, layer_num_heads,
                                       dropout, dropout, activation=F.elu,
                                       allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttentionTranslation(beta=torch.randn(1))
        self.num_meta_paths = num_meta_paths

        self.require_analyze = analyze

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)

    def analyze_node(self, g, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i].analyze_node(g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

    def analyze_meta_path(self, g, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(
                self.gat_layers[i].analyze_meta_path(g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

    @classmethod
    def from_original_model(cls, model, analyze=True, original_input=None,
                            original_label=None
                            ):
        """
        Create a new instance of HANLayer from the original model.
        """
        # we need to get the details of the original model by accessing its attributes
        in_size = model.gat_layers[-1]._in_src_feats
        out_size = model.gat_layers[-1]._out_feats
        layer_num_heads = model.gat_layers[-1]._num_heads
        dropout = model.gat_layers[-1].feat_drop.p
        model_use = cls(num_meta_paths=model.num_meta_paths,
                        in_size=in_size,
                        out_size=out_size,
                        layer_num_heads=layer_num_heads,
                        dropout=dropout,
                        analyze=analyze)
        # transfer the weights
        model_use.calculate_parameters(model, original_input, original_label)

        return model_use

    def calculate_parameters(self, model, original_input=None, original_label=None):
        """
        Calculate the parameters of the new model from the original model.
        """
        # self.semantic_attention.set_beta(model.semantic_attention.beta)
        # for i in range(self.num_meta_paths):
        #     self.gat_layers[i].calculate_parameters(model.gat_layers[i])
        if original_input is not None:
            # here we can calculate beta
            # beta = SemanticAttention.project(SemanticAttention_input).mean(0)
            # beta = torch.softmax(beta, dim=0)
            # self.semantic_attention.set_beta(beta)
            semantic_embeddings = []
            gs, h = original_input
            for i, g in enumerate(gs):
                semantic_embeddings.append(model.gat_layers[i](g, h).flatten(1))
            semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
            beta = model.semantic_attention.project(semantic_embeddings).mean(0)
            beta = torch.softmax(beta, dim=0)
            self.semantic_attention.set_beta(beta)

        # we just transfer the weights of the GAT layers
        for i in range(self.num_meta_paths):
            self.gat_layers[i].load_state_dict(model.gat_layers[i].state_dict())
