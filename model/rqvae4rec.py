import os
from pickle5 import pickle
from sklearn.cluster import KMeans
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from model.utils import *


class SEIDEmbeddingLoader(Dataset):
    def __init__(self, config):
        super().__init__()
        self.data = self.load_embedding(config['data_dir'], config['domain'], config['embedding_name'])
        self.dim = self.data[0].shape[0]

    def load_embedding(self, root_path, domain, name):
        with open(os.path.join(root_path, domain, f'{name}'), 'rb') as fIn:
            data = pickle.load(fIn)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = torch.from_numpy(self.data[idx])
        return idx, embedding

    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)

        input_idx = []
        input_emb = torch.zeros(B, self.dim)

        for i, entry in enumerate(batch):
            input_idx.append(entry[0])
            input_emb[i, :] = entry[1]

        batch_entry['item_idx'] = input_idx
        batch_entry['input_emb'] = input_emb

        return batch_entry


class CEIDEmbeddingLoader(Dataset):
    def __init__(self, config):
        super().__init__()
        self.data = self.load_embedding(config['data_dir'], config['domain'], config['embedding_name'])
        self.data_emb = self.data['item']
        self.dim = self.data_emb[0].shape[0]

    def load_embedding(self, root_path, domain, name):
        with open(os.path.join(root_path, domain, f'{name}'), 'rb') as fIn:
            data = pickle.load(fIn)
        return data

    def __len__(self):
        return len(self.data_emb)

    def __getitem__(self, idx):
        embedding = torch.from_numpy(self.data_emb[idx])
        return idx, embedding

    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)
        item_idx = []
        input_emb = torch.zeros(B, self.dim)
        for i, entry in enumerate(batch):
            item_idx.append(entry[0])
            input_emb[i, :] = entry[1]
        batch_entry['item_idx'] = item_idx
        batch_entry['input_emb'] = input_emb
        return batch_entry


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = [nn.Sequential(nn.Linear(config['dims'][i], config['dims'][i + 1]), nn.ReLU(True)) for i in range(config['n_layers'])]
        self.encoder = nn.Sequential(*layers)
        self.mapping = nn.Linear(config['dims'][-2], config['dims'][-1])
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        return self.mapping(self.encoder(x))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = [nn.Sequential(nn.Linear(config['dims'][-i - 1], config['dims'][-i - 2]), nn.ReLU(True)) for i in range(config['n_layers'])]
        self.decoder = nn.Sequential(*layers)
        self.mapping = nn.Linear(config['dims'][1], config['dims'][0])
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, z):
        return self.mapping(self.decoder(z))


def kmeans(samples, num_clusters, num_iters=100):
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()
    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)
    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)
    return tensor_centers


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, beta=0.25, kmeans_iter=10):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.kmeans_iter = kmeans_iter
        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.zero_()
        self.initialized = False

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_embedding(self, idx):
        return self.embedding(idx)

    def kmeans_init(self, data):
        centers = kmeans(data, self.n_embed, self.kmeans_iter)
        self.embedding.weight.data.copy_(centers)
        self.initialized = True

    def compute_distances(self, inputs):
        codebook_t = self.embedding.weight.t()
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == self.embed_dim
        inputs_flat = inputs.reshape(-1, self.embed_dim)
        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        return distances

    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)
        embed_idxs = distances.argmin(dim=-1)
        return embed_idxs

    def forward(self, z):
        if not self.initialized and self.training:
            self.kmeans_init(z)
        indices = self.find_nearest_embedding(z)
        z_q = self.embedding(indices)

        loss_commit = F.mse_loss(z_q.detach(), z)
        loss_codebook = F.mse_loss(z_q, z.detach())
        loss = loss_codebook + self.beta * loss_commit
        z_q = z + (z_q - z).detach()
        return z_q, loss, indices


class ResidualQuantizer(nn.Module):
    def __init__(self, code_len, n_embed, embed_dim, beta, kmeans_iter=100):
        super().__init__()
        self.code_len = code_len
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        codebooks = [VectorQuantizer(self.n_embed[idx], self.embed_dim, beta, kmeans_iter) for idx in range(self.code_len)]
        self.codebooks = nn.ModuleList(codebooks)

    def get_codebook(self):
        codebooks = []
        for quantizer in self.codebooks:
            codebook = quantizer.get_codebook
            codebooks.append(codebook)
        return torch.stack(codebooks)

    def forward(self, z):
        loss_all = []
        indices_all = []
        z_q = 0
        residual = z
        for quantizer in self.codebooks:
            z_out, loss, indices = quantizer(residual)
            residual = residual - z_out
            z_q = z_q + z_out
            loss_all.append(loss)
            indices_all.append(indices)
        loss_mean = torch.stack(loss_all).mean()
        indices_all = torch.stack(indices_all, dim=-1)
        return z_q, loss_mean, indices_all


class RQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quantizer = ResidualQuantizer(config['code_len'], config['n_embed'], config['embed_dim'], config['beta'], kmeans_iter=config['k_iter'])

    def forward(self, x):
        z_e = self.encode(x)
        z_q, quant_loss, code = self.quantizer(z_e)
        out = self.decode(z_q)
        return out, quant_loss, code

    def encode(self, x):
        z_e = self.encoder(x)
        return z_e

    def decode(self, z_q):
        out = self.decoder(z_q)
        return out

    @torch.no_grad()
    def get_codes(self, xs):
        z_e = self.encode(xs)
        _, _, code = self.quantizer(z_e)
        return code

    def compute_loss(self, out, quant_loss, code, x):
        loss_recon = F.mse_loss(out, x, reduction='mean')
        loss_latent = quant_loss
        loss_total = loss_recon + loss_latent

        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent,
            'codes': [code]
        }
