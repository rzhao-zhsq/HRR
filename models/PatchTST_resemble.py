# __all__ = ['PatchTST']

# Cell
import copy
from typing import Optional

from torch import Tensor
from torch import nn

from models.layers.PatchRevIN import RevIN
from models.layers.PatchTST_backbone_resemble import TSTEncoder, Flatten_Head
from models.layers.PatchTST_layers import positional_encoding


class Model(nn.Module):
    def __init__(
            self, cfg,
            fc_dropout=0.3, head_dropout=0.0, patch_len=16, stride=8, padding_patch="end",
            revin=1, affine=0, subtract_last=0, decomposition=0, kernel_size=25, individual=0,
            max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
            norm: str = 'BatchNorm', attn_dropout: float = 0., act: str = "gelu",
            key_padding_mask: bool = 'auto', padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None,
            res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
            pe: str = 'zeros', learn_pe: bool = True, pretrain_head: bool = False, head_type='flatten',
            verbose: bool = False, **kwargs
    ):

        super().__init__()

        # load parameters
        c_in = cfg.enc_in
        context_window = cfg.seq_len
        target_window = cfg.pred_len
        n_layers = cfg.e_layers
        n_heads = cfg.n_heads
        d_model = cfg.d_model
        d_ff = cfg.d_ff
        dropout = cfg.dropout

        # specifical parameters for PatchTST, overwrite original parameters
        fc_dropout = cfg.fc_dropout if hasattr(cfg, "fc_dropout") else fc_dropout
        head_dropout = cfg.head_dropout if hasattr(cfg, "head_dropout") else head_dropout
        patch_len = cfg.patch_len if hasattr(cfg, "patch_len") else patch_len
        stride = cfg.stride if hasattr(cfg, "stride") else stride
        padding_patch = cfg.padding_patch if hasattr(cfg, "padding_patch") else padding_patch
        revin = cfg.revin if hasattr(cfg, "revin") else revin
        affine = cfg.affine if hasattr(cfg, "affine") else affine
        subtract_last = cfg.subtract_last if hasattr(cfg, "subtract_last") else subtract_last
        # decomposition = cfg.decomposition if hasattr(cfg, "decomposition") else decomposition
        # kernel_size = cfg.kernel_size if hasattr(cfg, "kernel_size") else kernel_size
        individual = cfg.individual if hasattr(cfg, "individual") else individual

        # hyparameters
        self.diversity_weight = cfg.diversity_weight

        # compute patch num and patch length
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = (context_window - patch_len) // stride + 1
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.patch_num = patch_num
        self.d_model = d_model

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # token embedding and positional embedding
        self.seq_len = q_len = patch_num
        self.input_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = positional_encoding(pe, learn_pe, q_len, d_model)
        self.input_dropout = nn.Dropout(dropout) # Residual dropout

        # Encoder
        self.encoder = TSTEncoder(
            q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
            pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn,
            attn_type="sdp_resemble",
            sub_space=cfg.sub_space, sub_space_residual=cfg.sub_space_residual,
            out_space=cfg.out_space, out_space_residual=cfg.out_space_residual,
            diversity_metric=cfg.diversity_metric,
        )

        # predict head, for flatten and length adapt
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(
                self.head_nf, c_in, fc_dropout
            )  # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.head = Flatten_Head(
                self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout
            )


    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(head_nf, vars, 1)
        )

    def forward(self, x, *args):

        x = self.revin_layer(x, 'norm') if self.revin else x

        # patching
        x = x.permute(0, 2, 1)
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)                                 # [bs x nvars x seq_len + pad_size]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # [bs x nvars x patch_num x patch_len]

        # input embedding
        x = x.reshape(-1, self.patch_num, self.patch_len)                   # [bs * nvars x patch_num x patch_len]
        x = self.input_embedding(x)                                         # [bs * nvars x patch_num x d_model]
        x = self.input_dropout(x + self.pos_embedding)                      # [bs * nvars x patch_num x d_model]

        # Encoder
        encoder_output_dict = self.encoder(x)                               # [bs * nvars x patch_num x d_model]
        x = encoder_output_dict['output']
        x = x.reshape(-1, self.n_vars, self.patch_num, self.d_model)        # [bs x nvars x patch_num x d_model]
        x = x.permute(0,1,3,2)  # why permute?                              # [bs x nvars x d_model x patch_num]

        # predict head
        x = self.head(x)                                                    # [bs x nvars x target_window]
        x = x.permute(0, 2, 1)                                              # [bs x target_window x nvars]

        # denorm
        x = self.revin_layer(x, 'denorm') if self.revin else x

        return {
            "output": x,
            "diversity_loss": encoder_output_dict['diversity_loss'] * self.diversity_weight
        }

