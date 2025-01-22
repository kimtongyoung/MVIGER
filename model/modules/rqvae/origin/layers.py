import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans
from model.utils import activation


class MLPLayers(nn.Module):
    def __init__(self, dims, drop=0, act="relu", bn=False):
        super(MLPLayers, self).__init__()
        self.dims = dims
        self.drop = drop
        self.act = act
        self.bn = bn

        layers = []
        for idx, (inp_dim, out_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            layers.append(nn.Dropout(p=self.drop))
            layers.append(nn.Linear(inp_dim, out_dim))
            if self.bn:
                layers.append(nn.BatchNorm1d(out_dim))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers) - 2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name="relu", emb_dim=None):
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation


def kmeans(
        samples,
        num_clusters,
        num_iters=10,
):
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers
