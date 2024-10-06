from typing import Optional, Tuple, Union, Dict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import load_pretrained
from timm.models import register_model

from transformer_block import Block, get_sinusoid_encoding
from transformer import TransformerEncoder
from Sampling import CS_Sampling
from thop import profile
from torchsummary import summary


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        # Union类型的主要应用场景在于参数类型不确定或可选参数，即参数可以是多种不同的数据类型，可以灵活处理。
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        padding:Optional[Union[int, Tuple[int, int]]] = 0,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        assert isinstance(padding, Tuple)


        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class GlobalEncoder(nn.Module):
    """ Unfolded self attention"""
    def __init__(
        self,
        transformer_dim: int,
        mlp_ratio: int,
        n_transformer_blocks: int = 2,
        num_head: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        assert transformer_dim % num_head == 0

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                mlp_ratio=mlp_ratio,
                num_heads=num_head,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)


    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # in_channels=d
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C],P is patch_area
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:

        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:

        # convert feature map to patches
        # x:[B,C,H,W]
        patches, info_dict = self.unfolding(x)  # patches.shape:(B*patch_area,num_patches,C)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        return fm


class PyramidToken(nn.Module):
    """
    Hierarchical tokenization module
    """
    def __init__(self, img_size=224, tokens_type='convolution', in_chans=3, embed_dim=384, token_dim=64):
        super().__init__()

        if tokens_type == 'convolution':

            print('Adopt convolution + unfold attention for tokenization.')
            self.soft_split0 = nn.Sequential(ConvLayer(in_channels=in_chans,out_channels=token_dim,kernel_size=7,stride=4,padding=2),
                                             ConvLayer(in_channels=token_dim,out_channels=token_dim,kernel_size=1,stride=1,padding=0,
                                                       use_act=False,use_norm=False)
                                             )  # /4->(1,64,56,56)
            self.soft_split1 = nn.Sequential(
                ConvLayer(in_channels=token_dim, out_channels=token_dim, kernel_size=3, stride=2, padding=1),
                ConvLayer(in_channels=token_dim, out_channels=token_dim, kernel_size=1, stride=1, padding=0,
                          use_act=False, use_norm=False)
                )  # /2->(1,64,28,28)
            self.attention1 = GlobalEncoder(transformer_dim=token_dim,mlp_ratio=2,n_transformer_blocks=1,
                                            num_head=4,patch_h=4,patch_w=4)
            self.attention2 = GlobalEncoder(transformer_dim=token_dim,mlp_ratio=2,n_transformer_blocks=2,
                                            num_head=4,patch_h=2,patch_w=2)
            self.project = nn.Sequential(
                ConvLayer(in_channels=token_dim, out_channels=token_dim, kernel_size=3, stride=2, padding=1),
                ConvLayer(in_channels=token_dim, out_channels=embed_dim, kernel_size=1, stride=1, padding=0,
                          use_act=False, use_norm=False))  # /2->(1,384,14,14)
        else:
            pass

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: tokenization
        x = self.soft_split0(x)
        # iteration1: unfolded self-attention
        x = self.attention1(x)

        # step1: tokenization
        x = self.soft_split1(x)
        # iteration2: unfolded self-attention
        x = self.attention2(x)

        # step2: tokenization
        # /2->(1,384,14,14)->(1,196,384)
        x = self.project(x).flatten(2).transpose(1, 2)

        return x


class HTHA(nn.Module):
    def __init__(self, img_size=224, tokens_type='convolution',cs_ratio=0.1,blocksize=32, in_chans=3, num_classes=1000, embed_dim=384, depth=14,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.sample = CS_Sampling(n_channels=in_chans, cs_ratio=cs_ratio, blocksize=blocksize, im_size=img_size)
        self.tokens_to_token = PyramidToken(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth(drop path) decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.sample(x)
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def htha_14(img_size=224,tokens_type='convolution', cs_ratio=0.01,blocksize=32,pretrained_cfg=None,
                 pretrained_cfg_overlay=None,num_classes=1000,
                 embed_dim=384,depth=14,pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = HTHA(img_size=img_size,tokens_type=tokens_type, cs_ratio=cs_ratio,blocksize=blocksize,
                     embed_dim=embed_dim, depth=depth, num_heads=6, mlp_ratio=3., num_classes=num_classes,**kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def htha_24(img_size=224,tokens_type='convolution', cs_ratio=0.01,blocksize=32,pretrained_cfg=None,
                 pretrained_cfg_overlay=None,num_classes=1000,
                 embed_dim=512,depth=24,pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = HTHA(img_size=img_size,tokens_type=tokens_type, cs_ratio=cs_ratio,blocksize=blocksize,
                     embed_dim=embed_dim, depth=depth, num_heads=8, mlp_ratio=3., num_classes=num_classes,**kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


if __name__ == '__main__':
    inputs = torch.rand(1,3,384,384).cuda()
    model = htha_14(img_size=384,tokens_type='convolution', cs_ratio=0.1,blocksize=32).cuda()
    res = model(inputs)
    summary(model,input_size=(3,384,384),batch_size=1)
    macs,params = profile(model,inputs=(inputs,))
    print(f"Macs:{round(macs/(10**9),3)} G.")
    print(f"Params:{round(params/(10**6),3)} M.")
