# SPDX-License-Identifier: Apache-2.0
# Modified from TimeMixer; see NOTICE for attribution and changes.
"""LoadMixer forecasting model; multiscale backbone adapted from TimeMixer."""
from __future__ import annotations
import math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from layers import series_decomp, DataEmbedding_wo_pos, Normalize

class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList([nn.Sequential(torch.nn.Linear(configs.seq_len // configs.down_sampling_window ** i, configs.seq_len // configs.down_sampling_window ** (i + 1)), nn.GELU(), torch.nn.Linear(configs.seq_len // configs.down_sampling_window ** (i + 1), configs.seq_len // configs.down_sampling_window ** (i + 1))) for i in range(configs.down_sampling_layers)])

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = torch.nn.ModuleList([nn.Sequential(torch.nn.Linear(configs.seq_len // configs.down_sampling_window ** (i + 1), configs.seq_len // configs.down_sampling_window ** i), nn.GELU(), torch.nn.Linear(configs.seq_len // configs.down_sampling_window ** i, configs.seq_len // configs.down_sampling_window ** i)) for i in reversed(range(configs.down_sampling_layers))])

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))
        out_trend_list.reverse()
        return out_trend_list

class PastDecomposableMixing(nn.Module):

    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence
        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        else:
            raise ValueError('Only moving_avg decomposition is supported.')
        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(nn.Linear(in_features=configs.d_model, out_features=configs.d_ff), nn.GELU(), nn.Linear(in_features=configs.d_ff, out_features=configs.d_model))
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer = nn.Sequential(nn.Linear(in_features=configs.d_model, out_features=configs.d_ff), nn.GELU(), nn.Linear(in_features=configs.d_ff, out_features=configs.d_model))

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            (_, T, _) = x.size()
            length_list.append(T)
        season_list = []
        trend_list = []
        for x in x_list:
            (season, trend) = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        out_list = []
        for (ori, out_season, out_trend, length) in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs) for _ in range(configs.e_layers)])
        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.normalize_layers = torch.nn.ModuleList([Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False) for i in range(configs.down_sampling_layers + 1)])
        self.predict_layers = torch.nn.ModuleList([torch.nn.Linear(configs.seq_len // configs.down_sampling_window ** i, configs.pred_len) for i in range(configs.down_sampling_layers + 1)])
        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)
            self.out_res_layers = torch.nn.ModuleList([torch.nn.Linear(configs.seq_len // configs.down_sampling_window ** i, configs.seq_len // configs.down_sampling_window ** i) for i in range(configs.down_sampling_layers + 1)])
            self.regression_layers = torch.nn.ModuleList([torch.nn.Linear(configs.seq_len // configs.down_sampling_window ** i, configs.pred_len) for i in range(configs.down_sampling_layers + 1)])

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                (x_1, x_2) = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                (B, T, N) = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
        (x_enc, x_mark_enc) = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for (i, x, x_mark) in zip(range(len(x_enc)), x_enc, x_mark_enc):
                (B, T, N) = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for (i, x) in zip(range(len(x_enc)), x_enc):
                (B, T, N) = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for (i, x, x_mark) in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for (i, x) in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for (i, enc_out) in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            for (i, enc_out, out_res) in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        return dec_out_list

    def forward(self, x, x_mark=None, x_dec=None, y_mark=None, mask=None):
        return self.forecast(x, x_mark, x_dec, y_mark)

class PhaseAlignedStrideConvDownsampler(nn.Module):
    """Match non-overlapping average pooling at initialization, then learn."""

    def __init__(self, in_channels: int, kernel_size: int, stride: int, delta_scale: float):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=0, groups=in_channels, bias=False)
        nn.init.constant_(self.conv.weight, 1.0 / kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.delta_scale == 1.0:
            return self.conv(x)
        uniform = torch.full_like(self.conv.weight, 1.0 / self.conv.kernel_size[0])
        weight = uniform + self.delta_scale * (self.conv.weight - uniform)
        return F.conv1d(x, weight, stride=self.conv.stride, padding=0, groups=self.conv.groups)

class OrderedInnovationExpert(nn.Module):
    """Encode an ordered normalized change path and predict a level residual."""

    def __init__(self, input_size: int, width: int, pred_len: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(width, pred_len)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ordered_path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        content = self.encoder(ordered_path)
        return (content, torch.tanh(self.head(content)))

class LoadMixer(Model):
    """Univariate LoadMixer. Input [batch, 96, 1], output [batch, horizon, 1]."""

    def __init__(self, pred_len=12, d_model=16, d_ff=32, e_layers=2, dropout=0.1):
        configs = SimpleNamespace(task_name='long_term_forecast', seq_len=96, label_len=0, pred_len=pred_len, down_sampling_window=3, down_sampling_layers=1, channel_independence=1, d_model=d_model, d_ff=d_ff, e_layers=e_layers, dropout=dropout, decomp_method='moving_avg', moving_avg=25, enc_in=1, c_out=1, embed='timeF', freq='t', use_norm=1, use_future_temporal_feature=0)
        super().__init__(configs)
        self.scales = configs.down_sampling_layers + 1
        for layer in self.normalize_layers:
            layer.subtract_last = True
        self.scale_router = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, self.scales))
        nn.init.zeros_(self.scale_router[-1].weight)
        nn.init.zeros_(self.scale_router[-1].bias)
        self.scc_stride_convs = nn.ModuleList([PhaseAlignedStrideConvDownsampler(1, 3, 3, 1.0) for _ in range(configs.down_sampling_layers)])
        self._init_difference_correction(configs)

    def forecast(self, x, xm, xd, ym):
        self._routing_features = self.history(x)[0]
        return super().forecast(x, xm, xd, ym)

    def future_multi_mixing(self, B, enc_out_list, x_list):
        outs = super().future_multi_mixing(B, enc_out_list, x_list)
        weights = torch.softmax(self.scale_router(self._routing_features), dim=-1) * self.scales
        return [out * weights[:, :, i][:, None, :] for (i, out) in enumerate(outs)]

    def forward(self, x, x_mark=None):
        if x.ndim != 3 or x.shape[1:] != (96, 1):
            raise ValueError('Expected x with shape [batch, 96, 1].')
        if x_mark is not None and x_mark.shape != (x.shape[0], 96, 5):
            raise ValueError('Expected x_mark with shape [batch, 96, 5].')
        return super().forward(x, x_mark) + self._transient_correction(x)

    @staticmethod
    def history(x):
        scale = x.std(dim=1, keepdim=True, unbiased=False).detach().clamp_min(0.001)
        z = ((x - x.mean(dim=1, keepdim=True).detach()) / scale).detach()
        slopes = []
        for w in [6, 12, 24, 48, 96]:
            t = torch.arange(w, device=x.device, dtype=x.dtype)
            t = t - t.mean()
            slopes.append((z[:, -w:, :] * t[None, :, None]).sum(1) / (t * t).sum())
        features = torch.stack([z[:, -1], z[:, -6:].mean(1), z[:, -24:].mean(1), z[:, -6:].std(1, unbiased=False), z[:, -24:].std(1, unbiased=False), slopes[0] * 6, slopes[2] * 24, (z[:, 1:] - z[:, :-1]).abs().mean(1)], dim=-1)
        return (features, torch.stack(slopes, dim=-1), scale)

    def _Model__multi_scale_process_inputs(self, x, xm):
        xs = [x]
        marks = [xm] if xm is not None else None
        for conv in self.scc_stride_convs:
            x = conv(x.transpose(1, 2)).transpose(1, 2)
            xs.append(x)
            if xm is not None:
                xm = xm[:, ::self.configs.down_sampling_window, :][:, :x.shape[1], :]
                assert xm.shape[1] == x.shape[1]
                marks.append(xm)
        return (xs, marks)

    def _init_difference_correction(self, configs):
        width = int(getattr(configs, 'scc_transient_width', 0))
        if width <= 0:
            width = configs.d_model
        dropout = float(getattr(configs, 'scc_transient_dropout', -1.0))
        if dropout < 0:
            dropout = configs.dropout
        self.scc_transient_scale = float(getattr(configs, 'scc_transient_scale', 2.0))
        windows = getattr(configs, 'scc_transient_windows', None)
        if windows is None:
            windows = tuple((min(w, configs.seq_len - 1) for w in (6, 12, 24, 48, 95)))
        if not windows or any((int(w) != w or w < 1 or w >= configs.seq_len for w in windows)):
            raise ValueError('Difference windows must be integer lengths in [1, seq_len-1]')
        self.scc_transient_scales = tuple(sorted(set((int(w) for w in windows))))
        self.scc_transient_experts = nn.ModuleList([OrderedInnovationExpert(scale * configs.enc_in, width, self.pred_len, dropout) for scale in self.scc_transient_scales])
        state_size = 7 + len(self.scc_transient_scales) + width * len(self.scc_transient_scales)
        router_width = 2 * width
        self.scc_transient_router = nn.Sequential(nn.Linear(state_size, router_width), nn.GELU(), nn.Linear(router_width, len(self.scc_transient_scales) + 1))
        nn.init.zeros_(self.scc_transient_router[-1].weight)
        nn.init.zeros_(self.scc_transient_router[-1].bias)
        with torch.no_grad():
            self.scc_transient_router[-1].bias[0] = math.log(len(self.scc_transient_scales))

    @staticmethod
    def _base_state(x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        eps = 1e-05
        recent = min(12, dx.shape[1])
        x_recent = x[:, -recent:, :]
        dx_recent = dx[:, -recent:, :]
        level_mean = x.mean(dim=1)
        level_std = x.std(dim=1, unbiased=False).clamp_min(eps)
        long_rms = dx.square().mean(dim=1).add(eps).sqrt()
        recent_rms = dx_recent.square().mean(dim=1).add(eps).sqrt()
        features = [((x_recent.mean(dim=1) - level_mean) / level_std).mean(dim=1), ((x[:, -1, :] - level_mean) / level_std).mean(dim=1), ((x_recent[:, -1, :] - x_recent[:, 0, :]) / level_std).mean(dim=1), torch.log(recent_rms / long_rms.clamp_min(eps)).mean(dim=1), (F.relu(dx_recent).mean(dim=1) / long_rms).mean(dim=1), (F.relu(-dx_recent).mean(dim=1) / long_rms).mean(dim=1), (dx_recent.abs().amax(dim=1) / long_rms).mean(dim=1)]
        return torch.stack(features, dim=1)

    def _transient_correction(self, x_enc: torch.Tensor) -> torch.Tensor:
        eps = 1e-05
        dx = x_enc[:, 1:, :] - x_enc[:, :-1, :]
        global_rms = dx.square().mean(dim=(1, 2), keepdim=False).add(eps).sqrt()
        contents = []
        outputs = []
        scale_states = []
        for (scale, expert) in zip(self.scc_transient_scales, self.scc_transient_experts):
            window = dx[:, -scale:, :]
            local_rms = window.square().mean(dim=(1, 2), keepdim=False).add(eps).sqrt()
            ordered_path = (window / local_rms[:, None, None]).flatten(1)
            (content, normalized_residual) = expert(ordered_path)
            contents.append(content)
            outputs.append(normalized_residual * local_rms[:, None])
            scale_states.append(torch.log(local_rms / global_rms.clamp_min(eps)))
        route_state = torch.cat([self._base_state(x_enc, dx), torch.stack(scale_states, dim=1), *contents], dim=1)
        active_routes = torch.softmax(self.scc_transient_router(route_state), dim=1)[:, 1:]
        correction = (active_routes.unsqueeze(-1) * torch.stack(outputs, dim=1)).sum(dim=1)
        return (self.scc_transient_scale * correction).unsqueeze(-1)
