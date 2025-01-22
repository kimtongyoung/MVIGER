# use yonsei server
# from pickle5 import pickle
# use local
import pickle
from typing import Iterable
import numpy as np
import torch
import tqdm
import torch.distributed as dist
from torch.nn import functional as F
from model.utils import *
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import json
from torch.utils.data.dataset import Dataset
import os
from collections import defaultdict


class SentenceEmbeddingDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.datamap = self.load_json(os.path.join(config['data_dir'], config['domain'], 'rqid/datamaps.json'))[
            'id2item']
        self.meta_dict = self.load_json(os.path.join(config['data_dir'], config['domain'], 'rqid/meta_data.json'))
        self.item_keys = list(self.datamap.keys())
        self.model = SentenceTransformer(config['model_path']).cpu()
        self.tokenizer = AutoTokenizer.from_pretrained(config['backbone'], model_max_length=512)

    def __len__(self):
        return len(self.datamap)

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def __getitem__(self, idx):
        asin = self.datamap[str(idx)]
        meta = self.meta_dict[asin]
        # ['title', 'price', 'brand', 'categories']
        text = ''
        if 'title' in meta.keys():
            text += f'{meta["title"]}'
        if 'price' in meta.keys():
            text += f' {meta["price"]}'
        if 'brand' in meta.keys():
            text += f' {meta["brand"]}'
        if 'categories' in meta.keys():
            text += f' {meta["categories"][0]}'

        out_dict = defaultdict()
        out_dict['text'] = text
        embedding = self.model.encode(text)
        out_dict['embedding'] = embedding
        out_dict['idx'] = idx

        return out_dict


class SemIDEmbeddingLoader(Dataset):
    def __init__(self, config):
        super().__init__()
        self.data = self.load_embedding(config['data_dir'], config['domain'], config['sid_name'])
        self.dim = self.data[0].shape[0]

    def load_embedding(self, root_path, domain, name):
        with open(os.path.join(root_path, domain, f'rqid/rq-sid/{name}.pkl'), 'rb') as fIn:
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


class GIDEmbeddingLoader(Dataset):
    def __init__(self, config, user=False):
        super().__init__()
        self.user = user
        self.data = self.load_embedding(config['data_dir'], config['domain'])
        if self.user:
            self.data_emb = self.data['user_final_embed']
        else:
            self.data_emb = self.data['item_final_embed']
        self.dim = self.data_emb[0].shape[0]
    def load_embedding(self, root_path, domain):
        if self.user:
            with open(os.path.join(root_path, domain, 'rqid/rq-gid-user/embedding.pkl'), 'rb') as fIn:
                data = pickle.load(fIn)
        else:
            with open(os.path.join(root_path, domain, 'rqid/rq-gid/embedding.pkl'), 'rb') as fIn:
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


class HIDEmbeddingLoader(Dataset):
    def __init__(self, config, mix_type):
        super().__init__()
        self.sem_data = self.load_sem_embedding(config['data_dir'], config['domain'], config['sid_name'])
        self.g_data = self.load_g_embedding(config['data_dir'], config['domain'])['item_final_embed']
        self.mix_type = mix_type
        self.s_dim = self.sem_data[0].shape[0]
        self.g_dim = self.g_data[0].shape[0]

    def load_sem_embedding(self, root_path, domain, name):
        with open(os.path.join(root_path, domain, f'rqid/rq-sid/{name}.pkl'), 'rb') as fIn:
            data = pickle.load(fIn)
        return data

    def load_g_embedding(self, root_path, domain):
        with open(os.path.join(root_path, domain, 'rqid/rq-gid/embedding.pkl'), 'rb') as fIn:
            data = pickle.load(fIn)
        return data

    def __len__(self):
        return len(self.g_data)

    def __getitem__(self, idx):
        sem_embedding = torch.from_numpy(self.sem_data[idx])
        g_embedding = torch.from_numpy(self.g_data[idx])
        if self.mix_type == 'cat':
            embedding = torch.cat([sem_embedding, g_embedding], 0)
            return idx, embedding
        elif self.mix_type == 'avg':
            embedding = (sem_embedding + g_embedding) / 2
            return idx, embedding

    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)

        input_idx = []
        if self.mix_type == 'cat':
            input_emb = torch.zeros(B, self.s_dim +self.g_dim)
        else:
            input_emb = torch.zeros(B, self.g_dim)

        for i, entry in enumerate(batch):
            input_idx.append(entry[0])
            input_emb[i, :] = entry[1]

        batch_entry['item_idx'] = input_idx
        batch_entry['input_emb'] = input_emb

        return batch_entry


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['norm_type'] == 'bn':
            norm = nn.BatchNorm1d
        elif config['norm_type'] == 'ln':
            norm = nn.LayerNorm
        else:
            norm = nn.Identity
        if config['en_de_layer_num'] == 0:
            self.encoder = nn.Identity()
            self.mapping = nn.Linear(config['emb_dim'], config['latent_dim'])

        elif config['en_de_layer_num'] == -1:
            self.encoder = nn.Identity()
            self.mapping = nn.Identity()
        else:
            layers = [nn.Sequential(nn.Linear(config['encoder_channels'][i], config['encoder_channels'][i + 1]),
                                    norm(config['encoder_channels'][i + 1]),
                                    nn.ReLU(True),
                                    nn.Dropout(config['drop'])) for i in range(config['en_de_layer_num'])]
            self.encoder = nn.Sequential(*layers)
            self.mapping = nn.Linear(config['encoder_channels'][-1], config['latent_dim'])


    def forward(self, x):
        return self.mapping(self.encoder(x))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['norm_type'] == 'bn':
            norm = nn.BatchNorm1d
        elif config['norm_type'] == 'ln':
            norm = nn.LayerNorm
        else:
            norm = nn.Identity
        if config['en_de_layer_num'] == 0:
            self.decoder = nn.Identity()
            self.mapping = nn.Linear(config['latent_dim'], config['emb_dim'])
        elif config['en_de_layer_num'] == -1:
            self.decoder = nn.Identity()
            self.mapping = nn.Identity()
        else:
            layers = [nn.Sequential(nn.Linear(config['decoder_channels'][i], config['decoder_channels'][i + 1]),
                                    norm(config['decoder_channels'][i + 1]),
                                    nn.ReLU(True),
                                    nn.Dropout(config['drop'])) for i in range(config['en_de_layer_num'])]
            self.decoder = nn.Sequential(*layers)
            self.mapping = nn.Linear(config['decoder_channels'][-1], config['emb_dim'])



    def forward(self, z):
        return self.mapping(self.decoder(z))


class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)
        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed
        # exponential moving average to update codebook embeddings
        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]
            # padding index is not updated by EMA
            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))  # N
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())  # m

    @torch.no_grad()
    def compute_distances(self, inputs):
        # [size, dim] transpose -> [dim, size]
        codebook_t = self.weight[:-1, :].t()
        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        # origin: b, w, h, c -> b x w x h, c /// in our setting, (b,c)
        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        # inp + alpha(mat1 @ mat2)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, h, w, size]
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        # indexing by NN
        distances = self.compute_distances(inputs)  # [B, h, w, n_embed]
        embed_idxs = distances.argmin(dim=-1)
        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        # 빈 cluster 존재?
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        # [size, dim]
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]
        # b, c
        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        # b
        n_vectors = vectors.shape[0]
        # size
        n_total_embed = n_embed

        # size, num_features
        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0,
                              index=idxs.unsqueeze(0),
                              src=vectors.new_ones(1, n_vectors)
                              )
        # [size, num_features] -> multi hot encoding for counting clusters

        cluster_size = one_hot_idxs.sum(dim=1)
        # [size] -> counts
        vectors_sum_per_cluster = one_hot_idxs @ vectors
        # [size, num_features] @ [num_features, dim] = [size, dim]

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        # N_t = N_(t-1) * gamma + n_t * (1-gamma), gamma = 0.99
        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        # m_t = m_(t-1) * gamma + sum(vectors_per_cluster * (1-gamma))
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)

        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):
        # ema weight update
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
                n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        # 각 featuremap에 index 할당 [b,w,h]
        if self.training:
            if self.ema:
                # 버퍼갱신
                self._update_buffers(inputs, embed_idxs)
        embeds = self.embed(embed_idxs)
        if self.ema and self.training:
            # 임베딩 갱신
            self._update_embedding()
        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds


class RQBottleneck(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.

    Arguments:
        latent_shape (Tuple[int, int, int]): the shape of latents, denoted (H, W, D)
        code_shape (Tuple[int, int, int]): the shape of codes, denoted (h, w, d)
        n_embed (int, List, or Tuple): the number of embeddings (i.e., the size of codebook)
            If isinstance(n_embed, int), the sizes of all codebooks are same.
        shared_codebook (bool): If True, codebooks are shared in all location. If False,
            uses separate codebooks along the ``depth'' dimension. (default: False)
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(self,
                 n_embed,
                 embed_dim,
                 n_layers,
                 decay=0.99,
                 shared_codebook=False,
                 restart_unused_codes=True,
                 commitment_loss='cumsum'
                 ):
        super().__init__()
        self.shared_codebook = shared_codebook
        self.n_layers = n_layers
        # same codebook for each level -> False
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible \
                                    with list types of momentums or sizes: Change it into int")

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, Iterable) else [n_embed for _ in range(self.n_layers)]
        self.decay = decay if isinstance(decay, Iterable) else [decay for _ in range(self.n_layers)]
        assert len(self.n_embed) == self.n_layers
        assert len(self.decay) == self.n_layers

        if self.shared_codebook:
            codebook0 = VQEmbedding(self.n_embed[0],
                                    embed_dim,
                                    decay=self.decay[0],
                                    restart_unused_codes=restart_unused_codes,
                                    )
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.n_layers)])
        else:
            codebooks = [VQEmbedding(self.n_embed[idx],
                                     embed_dim,
                                     decay=self.decay[idx],
                                     restart_unused_codes=restart_unused_codes,
                                     ) for idx in range(self.n_layers)]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss

    def quantize(self, x):
        r"""
        Return list of quantized features and the selected codewords by the residual quantization.
        The split-gcn-code is selected by the residuals between x and quantized features by the previous codebooks.

        Arguments:
            x (Tensor): bottleneck feature maps to quantize.

        Returns:
            quant_list (list): list of sequentially aggregated and quantized feature maps by codebooks.
            codes (LongTensor): codewords index, corresponding to quants.

        Shape:
            - x: (B, h, w, embed_dim)
            - quant_list[i]: (B, h, w, embed_dim)
            - codes: (B, h, w, d)
        """
        # B, h, w, embed_dim = x.shape

        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.n_layers):
            quant, code = self.codebooks[i](residual_feature)
            # embeds, embed_idx

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        # return ([b, dim] x n_layers), [b, n_layers]
        return quant_list, codes

    def forward(self, x):
        quant_list, codes = self.quantize(x)

        commitment_loss = self.compute_commitment_loss(x, quant_list)
        # z\hat
        # quants_trunc = self.to_latent_shape(quant_list[-1])
        quants_trunc = x + (quant_list[-1] - x).detach()
        # z\hat, L_commitment, Sem_IDs
        return quants_trunc, commitment_loss, codes

    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []

        for idx, quant in enumerate(quant_list):
            partial_loss = (x - quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)

        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss

    @torch.no_grad()
    def embed_code(self, code):
        # embedding update 진행
        # [B, 1] x n_layers
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        # [b, n_layers, dim] -> [b, dim]
        embeds = torch.cat(embeds, dim=-2).sum(-2)
        # index to z_hat
        return embeds

    @torch.no_grad()
    def embed_code_with_depth(self, code, to_latent_shape=False):
        assert code.shape[-1] == self.n_layers
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        embeds = torch.cat(embeds, dim=-2)
        # [batch, n_layers, dim]
        return embeds, None

    @torch.no_grad()
    def embed_partial_code(self, code, code_idx, decode_type='select'):
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        if decode_type == 'select':
            embeds = embeds[code_idx]
        elif decode_type == 'add':
            embeds = torch.cat(embeds[:code_idx + 1], dim=-2).sum(-2)
        else:
            raise NotImplementedError(f"{decode_type} is not implemented in partial decoding")

        # embeds = self.to_latent_shape(embeds)

        return embeds

    @torch.no_grad()
    def get_soft_codes(self, x, temp=1.0, stochastic=False):
        # x = self.to_code_shape(x)
        residual_feature = x.detach().clone()
        soft_code_list = []
        code_list = []

        n_codebooks = self.n_layers
        for i in range(n_codebooks):
            codebook = self.codebooks[i]
            distances = codebook.compute_distances(residual_feature)
            soft_code = F.softmax(-distances / temp, dim=-1)

            if stochastic:
                soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
                code = torch.multinomial(soft_code_flat, 1)
                code = code.reshape(*soft_code.shape[:-1])
            else:
                code = distances.argmin(dim=-1)
            quants = codebook.embed(code)
            residual_feature -= quants

            code_list.append(code.unsqueeze(-1))
            soft_code_list.append(soft_code.unsqueeze(-2))

        code = torch.cat(code_list, dim=-1)
        soft_code = torch.cat(soft_code_list, dim=-2)
        return soft_code, code


class RQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.quantizer = RQBottleneck(
            n_embed=config['n_codebooks'],
            embed_dim=config['latent_dim'],
            n_layers=config['n_layers'],
            decay=config['decay'],
            shared_codebook=config['shared_codebook'],
            restart_unused_codes=config['restart_unused_codes'],
        )
        self.loss_type = 'mse'
        self.latent_loss_weight = config['latent_loss_weight']

    def forward(self, xs):
        z_e = self.encode(xs)
        z_q, quant_loss, code = self.quantizer(z_e)
        out = self.decode(z_q)
        return out, quant_loss, code

    def encode(self, x):
        z_e = self.encoder(x)
        # z_e = self.quant_conv(z_e).permute(0, 2, 3, 1).contiguous()
        return z_e

    def decode(self, z_q):
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()
        # z_q = self.post_quant_conv(z_q)
        out = self.decoder(z_q)
        return out

    @torch.no_grad()
    def get_codes(self, xs):
        z_e = self.encode(xs)
        _, _, code = self.quantizer(z_e)
        return code

    @torch.no_grad()
    def get_soft_codes(self, xs, temp=1.0, stochastic=False):
        assert hasattr(self.quantizer, 'get_soft_codes')

        z_e = self.encode(xs)
        soft_code, code = self.quantizer.get_soft_codes(z_e, temp=temp, stochastic=stochastic)
        return soft_code, code

    @torch.no_grad()
    def decode_code(self, code):
        z_q = self.quantizer.embed_code(code)
        decoded = self.decode(z_q)
        return decoded

    def compute_loss(self, out, quant_loss, code, xs=None, valid=False):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_latent = quant_loss
        loss_total = loss_recon + self.latent_loss_weight * loss_latent

        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent,
            'codes': [code]
        }

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def get_code_emb_with_depth(self, code):
        return self.quantizer.embed_code_with_depth(code)

    @torch.no_grad()
    def decode_partial_code(self, code, code_idx, decode_type='select'):
        r"""
        Use partial codebooks and decode the codebook features.
        If decode_type == 'select', the (code_idx)-th codebook features are decoded.
        If decode_type == 'add', the [0,1,...,code_idx]-th codebook features are added and decoded.
        """
        z_q = self.quantizer.embed_partial_code(code, code_idx, decode_type)
        decoded = self.decode(z_q)
        return decoded

    @torch.no_grad()
    def forward_partial_code(self, xs, code_idx, decode_type='select'):
        r"""
        Reconstuct an input using partial codebooks.
        """
        code = self.get_codes(xs)
        out = self.decode_partial_code(code, code_idx, decode_type)
        return out


def get_embedding(dset):
    result = defaultdict()
    for idx in tqdm.tqdm(range(len(dset))):
        result[dset[idx]['idx']] = dset[idx]['embedding']
    return result


def save_embedding(root_path, domain, result_dict, name='embedding'):
    with open(os.path.join(root_path, domain, f'rqid/rq-sid/{name}.pkl'), 'wb') as fOut:
        pickle.dump(result_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    return print('saving embedding done!')


def load_embedding(root_path, domain, name):
    with open(os.path.join(root_path, domain, f'rq-sid/{name}.pkl'), 'rb') as fIn:
        data = pickle.load(fIn)
    return data

# config = {
#     "backbone": "bert-large-cased",
#     "data_dir": "data/amazon/filtered",
#     "domain": "Beauty",
#     "model_path": "t5-small",
#     "en_de_layer_num": 3,
#     "sid_name": "t5-small-embedding",
#     "norm_type": 'none',
#     "encoder_channels": [
#         512,
#         256,
#         128,
#         64
#     ],
#     "decoder_channels": [
#         32,
#         64,
#         128,
#         256
#     ],
#     "emb_dim": 512,
#     "latent_dim": 32,
#     "n_codebooks": 256,
#     "n_layers": 3,
#     "decay": 0.99,
#     "latent_loss_weight": 0.25,
#     "act": "relu",
#     "shared_codebook": False,
#     "restart_unused_codes": True,
#     "drop": 0.1
# }
# #
# dset = SentenceEmbeddingDataset(config)
# res = get_embedding(dset)
# save_embedding(config['data_dir'], config['domain'], res, config['sid_name'])

# loader = SemIDEmbeddingLoader(config)
# loader2 = GIDEmbeddingLoader(config, user=False)
# loader3 = HIDEmbeddingLoader(config)
# model = RQVAE(config)
