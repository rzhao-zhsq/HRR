from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# from collections import OrderedDict
from models.layers.PatchTST_layers import Transpose, get_activation_fn, positional_encoding


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


# Cell
class TSTEncoder(nn.Module):
    def __init__(
            self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
            norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
            res_attention=False, n_layers=1, pre_norm=False, store_attn=False,
            **kwargs
    ):
        super().__init__()
        self.res_attention = res_attention

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                    norm=norm, attn_dropout=attn_dropout, dropout=dropout, activation=activation,
                    res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                    **kwargs
                ) for _ in range(n_layers)
            ]
        )

    def forward(
            self,
            src: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
    ):
        output = src
        scores = None
        diversity_loss = torch.tensor(0., device=src.device)

        for layer in self.layers:
            layer_output_dict = layer(
                output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            output, scores = layer_output_dict['output'], layer_output_dict['attn_scores']
            if layer_output_dict.get("sub_diversity_loss", None) is not None:
                diversity_loss += layer_output_dict.get("sub_diversity_loss")
            if layer_output_dict.get("out_diversity_loss", None) is not None:
                diversity_loss += layer_output_dict.get("out_diversity_loss")

        return {
            "output": output,
            "diversity_loss": diversity_loss,
        }


class TSTEncoderLayer(nn.Module):
    def __init__(
            self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
            norm='BatchNorm', attn_dropout=0., dropout=0., bias=True, activation="gelu",
            res_attention=False, pre_norm=False,
            **kwargs
    ):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.res_attention = res_attention
        self.attn_weight = None

        # Multi-Head attention
        self.attn_layer = MultiheadAttentionLayer(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention,
            **kwargs
        )

        # Add & Norm
        self.attn_dropout = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
            self,
            src: Tensor,
            prev: Optional[Tensor]=None,
            key_padding_mask: Optional[Tensor]=None,
            attn_mask: Optional[Tensor]=None
    ):

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)

        ## Multi-Head attention
        attn_dict = self.attn_layer(
            src, src, src, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        src2, attn_weights, attn_scores = attn_dict['output'], attn_dict['attn_weights'], attn_dict['attn_scores']

        if self.store_attn:
            self.attn_weight = attn_weights

        ## Add & Norm
        src = src + self.attn_dropout(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)

        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout

        if not self.pre_norm:
            src = self.norm_ffn(src)

        attn_dict['output'] = src
        return attn_dict


class MultiheadAttentionLayer(nn.Module):
    def __init__(
            self, d_model, n_heads,
            d_k=None, d_v=None, res_attention=False,
            attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False,
            **kwargs,
    ):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention

        attn_type = kwargs.get("attn_type", "sdp")
        if attn_type == "sdp":
            self.attn = ScaledDotProductAttention(
                d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa
            )
        elif attn_type == "sdp_resemble":
            self.attn = ScaledDotProductAttentionResemble(
                d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa,
                sub_space=kwargs.get("sub_space", False),
                sub_space_residual=kwargs.get("sub_space", False),
                out_space=kwargs.get("sub_space", False),
                out_space_residual=kwargs.get("sub_space", False),
                diversity_metric=kwargs.get("diversity_metric", "bvc"),
            )
        # Poject output
        self.out_projection = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model),
            nn.Dropout(proj_dropout)  # additional dropout.
        )

    def forward(
            self,
            Q: Tensor,
            K: Optional[Tensor] = None,
            V: Optional[Tensor] = None,
            prev: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None
    ):

        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        attn_dict = self.attn(
            q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        output, attn_weights, attn_scores = attn_dict['output'], attn_dict['attn_weights'], attn_dict['attn_scores']

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.out_projection(output)

        attn_dict['output'] = output
        return attn_dict


class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self attention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            prev: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None
    ):
        """
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        """

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional), for layer 1, prev is None.
        if self.res_attention and prev is not None:
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1)) # attn_weights   : [bs x n_heads x max_q_len x q_len]

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        return {
            "output": output,
            "attn_weights": attn_weights,
            "attn_scores": attn_scores if self.res_attention else None,
        }


class ScaledDotProductAttentionResemble(nn.Module):
    """
    Resemble attention with Bias-Variance-Covariance Decomposition from
    "An Ensemble-based Regularization Method for Multi-Head Attention, ICAI, 2021"
    """
    def __init__(
            self, d_model, n_heads,
            attn_dropout=0., res_attention=False, lsa=False,
            sub_space=False, sub_space_residual=False,
            out_space=False, out_space_residual=False,
            diversity_metric="bvc",
    ):
        super(ScaledDotProductAttentionResemble, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

        # for multi-head resemble
        self.diversity_metric = diversity_metric
        self.sub_space, self.sub_space_residual = sub_space, sub_space_residual
        self.out_space, self.out_space_residual = out_space, out_space_residual

    def forward(
            self,
            q: Tensor, k: Tensor, v: Tensor,
            prev: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None
    ):

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1)) # attn_weights   : [bs x n_heads x max_q_len x q_len]

        sub_d = out_d = None
        if self.sub_space_residual or self.sub_space:
            sub_d = self.get_disgreement(v.transpose(1,2))
            if self.sub_space_residual:
                v += sub_d.transpose(1,2)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.out_space_residual or self.out_space:
            out_d = self.get_disgreement(output)
            if self.out_space_residual:
                output += out_d

        return {
            "output": output,
            "attn_weights": attn_weights,
            "attn_scores": attn_scores if self.res_attention else None,
            "sub_diversity_loss": sub_d.mean() if self.sub_space else None,
            "out_diversity_loss": out_d.mean() if self.out_space else None,
        }

    def get_disgreement(self, x):
        if self.diversity_metric == "bvc":
            return self._bia_variance_covariance(x)
        elif self.diversity_metric == "df":
            return self._double_fault(x)
        elif self.diversity_metric == "vanilla":
            return self._vanilla_diversity(x)
        else:
            raise TypeError

    def _bia_variance_covariance(self, x):
        _, _, H, _ = x.shape
        raw_index = torch.arange(H).expand(H, -1).reshape(-1)
        off_index = torch.arange(H).expand(H, -1) + torch.arange(H).unsqueeze(1)
        off_index = off_index.reshape(-1) % H
        subtract = x - x.mean(0, keepdim=True)
        disgreement = torch.mul(subtract[:, :, raw_index, :], subtract[:, :, off_index, :]).mean(2, keepdim=True)
        return disgreement

    def _double_fault(self, x, threshold=0.5):
        _, _, H, _ = x.shape
        raw_index = torch.arange(H).expand(H, -1).reshape(-1)
        off_index = torch.arange(H).expand(H, -1) + torch.arange(H).unsqueeze(1)
        off_index = off_index.reshape(-1) % H
        y_hat = x.mean(dim=(0,1,3), keepdim=True)
        subtract = (x - y_hat) / y_hat
        # disgreement = torch.mul(x[:, :, raw_index, :], x[:, :, off_index, :]).mean(2, keepdim=True)
        disgreement = torch.mul(x[:, :, raw_index, :], x[:, :, off_index, :]).mean(2, keepdim=True)

        return disgreement

    def _vanilla_diversity(self, x):
        _, _, H, _ = x.shape
        raw_index = torch.arange(H).expand(H, -1).reshape(-1)
        off_index = torch.arange(H).expand(H, -1) + torch.arange(H).unsqueeze(1)
        off_index = off_index.reshape(-1) % H
        disgreement = torch.pow(x[:, :, raw_index, :] - x[:, :, off_index, :], 2).mean(2, keepdim=True)
        return disgreement



