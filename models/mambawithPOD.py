import torch
import numpy as np
from sklearn.decomposition import PCA


import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple

from torch import Tensor
from typing import Optional

import torch
from functools import partial

from models.fno import FNO2d
from mamba_ssm.modules.mamba_simple import Mamba,MambawithfixedC_time,MambawithfixedC_space
from rope import *

from timm.models.layers import trunc_normal_, lecun_normal_

import random
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm,layer_norm_fn,rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# 切小方块的操作

# B C H W -> B embed_dim grid_size, grid_size -> B embed_dim num_patches

class Block(nn.Module):
    def __init__(
        self, d_model,d_state, modes,mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        super(Block, self).__init__()
        # Q4: 这个参数起到什么意思？下面几个参数

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(d_model,d_state,modes)            # 这个是在参数里面定义的
        self.norm = norm_cls(d_model)              # 这个是在参数里面定义的 self.norm = nn.LayerNorm(dim)
        # Q5: timm中的 droppath是什么意思？
        # A: 这个是在timm里面制定的,类似于dropout的一种方法，区别在于drop_path丢掉一个层
        self.drop_path = DropPath(drop_path)

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            # Q6: isinstance是什么意思？
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self,
                hidden_states: Tensor, residual: Optional[Tensor] = None,
                inference_params=None):
        # 这个block接受两个输入，分别是hidden_states, residual(可选)
        if not self.fused_add_norm:  # self.fused_add_norm表示是否要使用混合相加标准化的方式
            if residual is None:
                residual = hidden_states  # residual的实质其实是上一状态的输出
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

        else:  # 在没装好的时候需要使用下面的方法来装入
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:  # 当使用残差链接的时候
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),  # 唯一的区别是hidden_states需要用drop_path丢掉
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block_time(
        d_model,
        d_state,
        modes,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type="v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device":device, "dtype":dtype}
    mixer_cls = lambda dim,d_state,modes: MambawithfixedC_time(d_model=d_model, d_model_out=d_model,d_state=d_state,modes=modes)
    norm_cls=partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = Block(
        d_model=d_model,
        d_state=d_state,
        modes =modes,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block
class finalBlock(nn.Module):
    def __init__(
        self, d_model,d_state,d_model_out, mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        super(finalBlock, self).__init__()
        # Q4: 这个参数起到什么意思？下面几个参数

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(d_model,d_state,d_model_out)            # 这个是在参数里面定义的
        self.norm = norm_cls(d_model)              # 这个是在参数里面定义的 self.norm = nn.LayerNorm(dim)
        # Q5: timm中的 droppath是什么意思？
        # A: 这个是在timm里面制定的,类似于dropout的一种方法，区别在于drop_path丢掉一个层
        self.drop_path = DropPath(drop_path)

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            # Q6: isinstance是什么意思？
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self,
                hidden_states: Tensor, residual: Optional[Tensor] = None,
                inference_params=None):
        # 这个block接受两个输入，分别是hidden_states, residual(可选)
        if not self.fused_add_norm:  # self.fused_add_norm表示是否要使用混合相加标准化的方式
            if residual is None:
                residual = hidden_states  # residual的实质其实是上一状态的输出
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

        else:  # 在没装好的时候需要使用下面的方法来装入
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:  # 当使用残差链接的时候
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),  # 唯一的区别是hidden_states需要用drop_path丢掉
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
def create_finalblock_time(
        d_model,
        d_state,
        d_model_out,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type="v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device":device, "dtype":dtype}
    mixer_cls = lambda dim,d_state,d_model_out: MambawithfixedC_time(d_model=d_model, d_model_out=d_model_out,d_state=d_state)
    norm_cls=partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = finalBlock(
        d_model=d_model,
        d_state=d_state,
        d_model_out=d_model_out,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block

# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 16:21
# @Author  : lil louis
# @Location: Beijing
# @File    : Vim.py


def create_block_space(
        d_model,
        d_state,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type="v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device":device, "dtype":dtype}
    mixer_cls = lambda dim,d_state: MambawithfixedC_space(d_model=d_model, d_model_out=d_model,d_state=d_state)
    norm_cls=partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = Block(
        d_model=d_model,
        d_state=d_state,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block
class finalBlock(nn.Module):
    def __init__(
        self, d_model,d_state,d_model_out, mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        super(finalBlock, self).__init__()
        # Q4: 这个参数起到什么意思？下面几个参数

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(d_model,d_state,d_model_out)            # 这个是在参数里面定义的
        self.norm = norm_cls(d_model)              # 这个是在参数里面定义的 self.norm = nn.LayerNorm(dim)
        # Q5: timm中的 droppath是什么意思？
        # A: 这个是在timm里面制定的,类似于dropout的一种方法，区别在于drop_path丢掉一个层
        self.drop_path = DropPath(drop_path)

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            # Q6: isinstance是什么意思？
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self,
                hidden_states: Tensor, residual: Optional[Tensor] = None,
                inference_params=None):
        # 这个block接受两个输入，分别是hidden_states, residual(可选)
        if not self.fused_add_norm:  # self.fused_add_norm表示是否要使用混合相加标准化的方式
            if residual is None:
                residual = hidden_states  # residual的实质其实是上一状态的输出
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

        else:  # 在没装好的时候需要使用下面的方法来装入
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:  # 当使用残差链接的时候
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),  # 唯一的区别是hidden_states需要用drop_path丢掉
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
def create_finalblock_space(
        d_model,
        d_state,
        d_model_out,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type="v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device":device, "dtype":dtype}
    mixer_cls = lambda dim,d_state,d_model_out: MambawithfixedC_space(d_model=d_model, d_model_out=d_model_out,d_state=d_state)
    norm_cls=partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = finalBlock(
        d_model=d_model,
        d_state=d_state,
        d_model_out=d_model_out,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block

class MambaPOD_time(nn.Module):
    def __init__(self,
                 num_blocks=24,
                 d_model=192,
                 d_model_out=1000,
                 d_state =512,
                 wodth =12,
                 modes=16,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon:float=1e-5,
                 rms_norm:bool=False,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 if_flip_img_sequences=True,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,
                 **kwargs):
        factory_kwargs = {"device":device, "dtype":dtype}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_rope = if_rope
        self.if_flip_img_sequences =if_flip_img_sequences
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio

        self.d_model_out = d_model_out
        self.d_model = d_model
        self.d_state = d_state


        # drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate >0. else nn.Identity()
        self.layers = nn.ModuleList(
            [
                create_block_time(
                    d_model,
                    d_state,
                    modes,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs
                )
                for i in range(num_blocks-1)
            ]
        )


        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        self.finalBlock = create_finalblock_time(
            d_model,
            d_state,
            d_model_out,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=num_blocks,
            if_bimamba=if_bimamba,
            bimamba_type=bimamba_type,
            drop_path=inter_dpr[num_blocks],
            if_devide_out=if_devide_out,
            init_layer_scale=init_layer_scale,
            **factory_kwargs
        )

    def forward_features(self, x, inference_params=None,
                         if_random_cls_token_position=False,
                         if_random_token_rank=False):

        # mamba
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if self.if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params = inference_params
                )

        else:
            for i in range(len(self.layers) // 2):
               if self.if_rope:
                   hidden_states = self.rope(hidden_states)
                   if residual is not None and self.if_rope_residual:
                       residual = self.rope(residual)

               hidden_states_f, residual_f = self.layers[i*2](
                    hidden_states, residual, inference_params=inference_params
               )
               hidden_states_b, residual_b = self.layer[i*2 + 1](
                   hidden_states.flip([1]),
                   None if residual == None else residual.flip([1]),
                   inference_params=inference_params
               )
               hidden_states = hidden_states_f + hidden_states_b.flip([1])
               residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                residual_in_fp32=self.residual_in_fp32)


        return hidden_states,residual


    def forward(self, x,
               inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        hidden_states,residual = self.forward_features(x, inference_params,
                                  if_random_cls_token_position=if_random_cls_token_position,
                                  if_random_token_rank=if_random_token_rank)
        out,_ = self.finalBlock(hidden_states,residual)

        return out

class MambaPOD_space(nn.Module):
    def __init__(self,
                 num_blocks=24,
                 d_model=192,
                 d_model_out=1000,
                 d_state =512,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon:float=1e-5,
                 rms_norm:bool=False,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 if_flip_img_sequences=True,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,
                 **kwargs):
        factory_kwargs = {"device":device, "dtype":dtype}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_rope = if_rope
        self.if_flip_img_sequences =if_flip_img_sequences
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio

        self.d_model_out = d_model_out
        self.d_model = d_model
        self.d_state = d_state


        # drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate >0. else nn.Identity()
        self.layers = nn.ModuleList(
            [
                create_block_space(
                    d_model,
                    d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs
                )
                for i in range(num_blocks-1)
            ]
        )


        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        self.finalBlock = create_finalblock_space(
            d_model,
            d_state,
            d_model_out,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=num_blocks,
            if_bimamba=if_bimamba,
            bimamba_type=bimamba_type,
            drop_path=inter_dpr[num_blocks],
            if_devide_out=if_devide_out,
            init_layer_scale=init_layer_scale,
            **factory_kwargs
        )

    def forward_features(self, x, inference_params=None,
                         if_random_cls_token_position=False,
                         if_random_token_rank=False):

        # mamba
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if self.if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params = inference_params
                )

        else:
            for i in range(len(self.layers) // 2):
               if self.if_rope:
                   hidden_states = self.rope(hidden_states)
                   if residual is not None and self.if_rope_residual:
                       residual = self.rope(residual)

               hidden_states_f, residual_f = self.layers[i*2](
                    hidden_states, residual, inference_params=inference_params
               )
               hidden_states_b, residual_b = self.layer[i*2 + 1](
                   hidden_states.flip([1]),
                   None if residual == None else residual.flip([1]),
                   inference_params=inference_params
               )
               hidden_states = hidden_states_f + hidden_states_b.flip([1])
               residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                residual_in_fp32=self.residual_in_fp32)


        return hidden_states,residual


    def forward(self, x,
               inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        hidden_states,residual = self.forward_features(x, inference_params,
                                  if_random_cls_token_position=if_random_cls_token_position,
                                  if_random_token_rank=if_random_token_rank)
        out,_ = self.finalBlock(hidden_states,residual)

        return out

class  MambaPOD_time_FNO(nn.Module):
    def __init__(self,
                 modes1=80,
                 modes2=80,
                 modes=16,
                 width=12,
                 num_blocks=24,
                 d_model=16,
                 d_model_out=384*199,
                 d_state =32,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon:float=1e-5,
                 rms_norm:bool=False,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 if_flip_img_sequences=True,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,

                 **kwargs):
        factory_kwargs = {"device":device, "dtype":dtype}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_rope = if_rope
        self.if_flip_img_sequences =if_flip_img_sequences
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio

        self.d_model_out = d_model_out
        self.d_model = d_model
        self.d_state = d_state


        # drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate >0. else nn.Identity()
        self.layers = nn.ModuleList(
            [
                create_block_time(
                    d_model,
                    d_state,
                    modes,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs
                )
                for i in range(num_blocks-1)
            ]
        )


        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        self.finalBlock = create_finalblock_time(
            d_model,
            d_state,
            d_model_out,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=num_blocks,
            if_bimamba=if_bimamba,
            bimamba_type=bimamba_type,
            drop_path=inter_dpr[num_blocks],
            if_devide_out=if_devide_out,
            init_layer_scale=init_layer_scale,
            **factory_kwargs
        )
        self.FNO_final= FNO2d(modes1,modes2,width)

    def forward_features(self, x, inference_params=None,
                         if_random_cls_token_position=False,
                         if_random_token_rank=False):

        # mamba
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if self.if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params = inference_params
                )

        else:
            for i in range(len(self.layers) // 2):
               if self.if_rope:
                   hidden_states = self.rope(hidden_states)
                   if residual is not None and self.if_rope_residual:
                       residual = self.rope(residual)

               hidden_states_f, residual_f = self.layers[i*2](
                    hidden_states, residual, inference_params=inference_params
               )
               hidden_states_b, residual_b = self.layer[i*2 + 1](
                   hidden_states.flip([1]),
                   None if residual == None else residual.flip([1]),
                   inference_params=inference_params
               )
               hidden_states = hidden_states_f + hidden_states_b.flip([1])
               residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                residual_in_fp32=self.residual_in_fp32)


        return hidden_states,residual


    def forward(self, x,
               inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        hidden_states,residual = self.forward_features(x, inference_params,
                                  if_random_cls_token_position=if_random_cls_token_position,
                                  if_random_token_rank=if_random_token_rank)
        out,_ = self.finalBlock(hidden_states,residual)
        out = out.view(-1,out.size(1),384,199)
        out = out.permute(0, 2, 3, 1)
        _,_,_,c = out.size()
        outputs = []
        for i in range(c):
            out_slice = out[:,:,:,i:i+1]
            output = self.FNO_final(out_slice)
            outputs.append(output)


        out = torch.cat(outputs,dim=-1)
        out = out.permute(0, 3, 1, 2)
        out = out.flatten(2)

        return out


if __name__ == "__main__":
    device = torch.device("cuda")
    x = torch.randn(1,120, 16).to(device)
    model = MambaPOD_time_FNO().to(device)
    paranum = sum(a.numel() for a in model.parameters() if a.requires_grad)

    print(f"Number of trainable parameters: {paranum}")
    y = model(x)
    print(y.shape)



