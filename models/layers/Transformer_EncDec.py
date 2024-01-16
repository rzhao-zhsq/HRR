import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        attn_dict = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )

        # new_x, attn = self.attention(
        #     x, x, x,
        #     attn_mask=attn_mask
        # )
        # x = x + self.dropout(new_x)

        x = x + self.dropout(attn_dict['output'])
        y = x = self.norm1(x)
        y = self.conv1(y.transpose(-1, 1))
        y = self.dropout(self.activation(y))
        y = self.conv2(y).transpose(-1, 1)
        y = self.dropout(y)

        attn_dict['output'] = self.norm2(x + y)
        return attn_dict


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        sub_bs = []
        out_bs = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                attn_dict = attn_layer(x, attn_mask=attn_mask)
                x = attn_dict['output']
                attns.append(attn_dict['output'])
                sub_bs.append(attn_dict.get('sub_disagreement', None))
                out_bs.append(attn_dict.get('out_disagreement', None))

        if self.norm is not None:
            x = self.norm(x)

        return {
            "output": x,
            "attns": attns,
            "sub_disagreements": sub_bs,
            "out_disagreements": out_bs,
        }


class DecoderLayer(nn.Module):
    def __init__(
            self, self_attention, cross_attention, d_model, d_ff=None,dropout=0.1, activation="relu"
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        attn_dict = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )
        x = x + self.dropout(attn_dict['output'])
        x = self.norm1(x)

        cross_attn_dict = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )
        x = x + self.dropout(cross_attn_dict['output'])

        y = x = self.norm2(x)
        y = self.conv1(y.transpose(-1, 1))
        y = self.dropout(self.activation(y))
        y = self.conv2(y).transpose(-1, 1)
        y = self.dropout(y)
        y = self.norm3(x + y)

        return {
            "output": y,
            "attn": attn_dict['attn'],
            "cross_attn": cross_attn_dict['attn'],
            "sub_disagreement": attn_dict.get('sub_disagreement', None),
            "out_disagreement": attn_dict.get('out_disagreement', None),
            "cross_sub_disagreement": cross_attn_dict.get('sub_disagreement', None),
            "cross_out_disagreement": cross_attn_dict.get('out_disagreement', None),
        }


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        attns = []
        cross_attns = []
        sub_bs = []
        out_bs = []
        cross_sub_bs = []
        cross_out_bs = []
        for layer in self.layers:
            attn_dict = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            # x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            x = attn_dict['output']
            attns.append(attn_dict['attn'])
            sub_bs.append(attn_dict['sub_disagreement'])
            out_bs.append(attn_dict['out_disagreement'])
            cross_attns.append(attn_dict['cross_attn'])
            cross_sub_bs.append(attn_dict['cross_sub_disagreement'])
            cross_out_bs.append(attn_dict['cross_out_disagreement'])

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return {
            "output": x,
            "attns": attns,
            "sub_disagreements": sub_bs,
            "out_disagreements": out_bs,
            "cross_attns": cross_attns,
            "cross_sub_disagreements": cross_sub_bs,
            "cross_out_disagreements": cross_out_bs,
        }
