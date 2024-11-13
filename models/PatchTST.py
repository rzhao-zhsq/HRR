# __all__ = ['PatchTST']

# Cell
from typing import Optional

from torch import Tensor
from torch import nn

from models.layers.PatchTST_backbone import PatchTST_backbone
from models.layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(
            self, cfg,
            fc_dropout=0.3, head_dropout = 0.0, patch_len=16, stride=8, padding_patch="end",
            revin=1, affine=1, subtract_last=0,decomposition=0,kernel_size=25,individual=0,
            max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
            norm: str = 'BatchNorm', attn_dropout: float = 0., act: str = "gelu",
            key_padding_mask: bool = 'auto', padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None,
            res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
            pe: str = 'zeros', learn_pe: bool = True, pretrain_head: bool = False, head_type='flatten',
            verbose: bool = False, **kwargs
    ):
        super().__init__()
        if cfg.label_len != 0:
            print(f"For PatchTST, label length must be 0, automatically set from {cfg.label_len} to 0 now!")
            cfg.label_len = 0

        # load parameters
        c_in = cfg.enc_in
        context_window = cfg.seq_len
        target_window = cfg.pred_len
        n_layers = cfg.e_layers
        n_heads = cfg.n_heads
        d_model = cfg.d_model
        d_ff = cfg.d_ff
        dropout = cfg.dropout

        # specifical parameters for PatchTST
        fc_dropout = cfg.fc_dropout = cfg.fc_dropout if hasattr(cfg, "fc_dropout") else fc_dropout
        head_dropout = cfg.head_dropout = cfg.head_dropout if hasattr(cfg, "head_dropout") else head_dropout
        patch_len = cfg.patch_len = cfg.patch_len if hasattr(cfg, "patch_len") else patch_len
        stride = cfg.stride = cfg.stride if hasattr(cfg, "stride") else stride
        padding_patch = cfg.padding_patch = cfg.padding_patch if hasattr(cfg, "padding_patch") else padding_patch
        revin = cfg.revin = cfg.revin if hasattr(cfg, "revin") else revin
        affine = cfg.affine = cfg.affine if hasattr(cfg, "affine") else affine
        subtract_last = cfg.subtract_last = cfg.subtract_last if hasattr(cfg, "subtract_last") else subtract_last
        decomposition = cfg.decomposition = cfg.decomposition if hasattr(cfg, "decomposition") else decomposition
        kernel_size = cfg.kernel_size = cfg.kernel_size if hasattr(cfg, "kernel_size") else kernel_size
        individual = cfg.individual = cfg.individual if hasattr(cfg, "individual") else individual


        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride,
                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type,
                individual=individual, revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=verbose, **kwargs
            )
            self.model_res = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride,
                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=verbose, **kwargs
            )
        else:
            self.model = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride,
                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=verbose, **kwargs
            )

        self.loginfo = self._set_loginfo_(cfg)

    def _set_loginfo_(self, cfg):

        return f"_sl_{cfg.seq_len}_ll_{cfg.label_len}_pl_{cfg.pred_len}" \
               f"-freq_{cfg.freq}" \
               f"-freq_{cfg.revin}" \
               f"_embed_{cfg.embed}_kernel_{cfg.kernel_size}_encin_{cfg.enc_in}" \
               f"_el_{cfg.e_layers}_dmodel_{cfg.d_model}_nhead_{cfg.n_heads}_dff_{cfg.d_ff}" \
               f"_patchlen_{cfg.patch_len}_stride_{cfg.stride}" \
               f"_dropout_{cfg.dropout}_fcdropout_{cfg.fc_dropout}" \
               f"_decompos_{cfg.decomposition}_individual_{cfg.individual}"

    def get_loginfo(self):
        return self.loginfo

    # def forward(self, x):  # x: [Batch, Input length, Channel]
    def forward(self, target_x, **kwargs,):
        x = target_x
        if self.decomposition:
            res_init, trend_init = self.decomp_module.forward(x)
            # x: [Batch, Channel, Input length]
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model.forward(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Target length, Channel]
        return {
            "output": x,
        }


