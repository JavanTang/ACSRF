# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
# from tools.paint import paint_pool_feature
from mmcv.ops import DeformConv2dPack as DCN
import math

# Code for "Context-Gated Convolution"
# ECCV 2020
# Xudong Lin*, Lin Ma, Wei Liu, Shih-Fu Chang
# {xudong.lin, shih.fu.chang}@columbia.edu, forest.linma@gmail.com, wl2223@columbia.edu

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np  

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if kernel_size == 1:
            self.ind = True
        else:
            self.ind = False            
            self.oc = out_channels
            self.ks = kernel_size
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = padding
            self.stride = stride

            
            # the target spatial size of the pooling layer
            ws = kernel_size
            self.avg_pool = nn.AdaptiveAvgPool2d((ws,ws))
            
            # the dimension of the latent repsentation
            self.num_lat = int((kernel_size * kernel_size) / 2 + 1)
            
            # the context encoding module
            self.ce = nn.Linear(ws*ws, self.num_lat, False)            
            self.ce_bn = nn.BatchNorm1d(in_channels)
            self.ci_bn2 = nn.BatchNorm1d(in_channels)
            
            # activation function is relu
            self.act = nn.ReLU(inplace=True)
            
            
            # the number of groups in the channel interacting module
            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels
            # the channel interacting module     下面这个有bug  48 96 144 192
            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            self.ci_bn = nn.BatchNorm1d(out_channels)
            
            # the gate decoding module
            self.gd = nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            self.gd2 = nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            
            # used to prrepare the input feature map to patches
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)
            
            # sigmoid function
            self.sig = nn.Sigmoid()
    def forward(self, x):
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if self.ind:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size() # torch.Size([1, 96, 128, 128])
            weight = self.weight
            # allocate glbal information
            gl = self.avg_pool(x).view(b,c,-1) # [1,96,4]  4 为 kernel*kernel
            # paint_pool_feature(gl, 'gl.jpg')
            # context-encoding module
            out = self.ce(gl)  # nn.Linear(ws*ws, self.num_lat, False)  self.num_lat = int((kernel_size * kernel_size) / 2 + 1)
            # use different bn for the following two branches
            ce2 = out
            out = self.ce_bn(out)
            out = self.act(out)
            # gate decoding branch 1
            out = self.gd(out) # nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            # channel interacting module
            if self.g >3:
                # grouped linear   self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
                oc = self.ci(self.act(self.ci_bn2(ce2).\
                                      view(b, c//self.g, self.g, -1).transpose(2,3))).transpose(2,3).contiguous()
            else:
                # linear layer for resnet.conv1
                oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2,1))).transpose(2,1).contiguous()  # nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            oc = oc.view(b,self.oc,-1) # torch.Size([1, 6, 21, 3]) 1, 378 -> 1, 128 , 3
            oc = self.ci_bn(oc)
            oc = self.act(oc)
            # gate decoding branch 2
            oc = self.gd2(oc)    # nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            # produce gate
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))
            # unfolding input feature map to patches
            x_un = self.unfold(x)
            b, _, l = x_un.size()
            # gating
            out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)
            # currently only handle square input and output
            h_out = math.floor((h + 2 * self.padding - self.kernel_size[0]) / self.stride) + 1
            w_out = math.floor((w + 2 * self.padding - self.kernel_size[0]) / self.stride) + 1

            return torch.matmul(out,x_un).view(b, self.oc, h_out, w_out)   



class DCNBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DCNBlock, self).__init__()
        self.dcn = DCN(in_channel, out_channel, kernel_size=(3,3), stride=1, padding=1)
    def forward(self, x):
        return self.dcn(x)

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers  通道信息融合
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mix(nn.Module):
    # 解决低级特征传递到高级特征的信息传递问题,这里直接使用的是UNet的做法
    def __init__(self, inchannel=64):
        super(Mix, self).__init__()
        self.conv = Conv2d(inchannel*2, inchannel*2, kernel_size=1, stride=1, padding=0, bias=False)

        self.ck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(inchannel*2, inchannel),
            nn.Linear(inchannel,inchannel*2),
            )

        self.conv2 = Conv2d(inchannel*2, inchannel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, fea1, fea2):
        # print(fea1.shape, fea2.shape)
        out = torch.cat([fea1, fea2], dim=1)
        out = self.conv(out)

        ck_out = self.ck(out)
        scale = torch.sigmoid(ck_out).unsqueeze(2).unsqueeze(3).expand_as(out)

        out = out * scale
        out = self.conv2(out)
        
        return out

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0. 0.5 在dence效果最好,
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, out_chans=3, num_classes=1000, 
                 depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], drop_path_rate=0.5, 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        self.mixs = nn.ModuleList()

        # ----上采样----
        self.stem = nn.Sequential(
            Conv2d(in_chans, dims[0], kernel_size=5,padding=2),
            nn.ReLU(True)
        )
        
        self.downsample_layers.append(self.stem)
        # 508
        self.downsample_layers.append(nn.Sequential(
                    Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
                    nn.ReLU(True)
            ))
        # 254
        self.downsample_layers.append(nn.Sequential(
                    Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
                    nn.ReLU(True)
        ))
        # 127
        self.downsample_layers.append(nn.Sequential(
                    Conv2d(dims[2], dims[3], kernel_size=2, stride=2),
                    nn.ReLU(True)
        ))
        # 63

        self.dcnbs = nn.ModuleList()
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            self.dcnbs.append(DCNBlock(dims[i], dims[i]))
            cur += depths[i]

        # ----下采样----
        self.upsample_layers = nn.ModuleList() # 3 intermediate upsampling conv layers
        self.upsample_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2),
                    nn.ReLU(True)
            ))
        self.upsample_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2),
                    nn.ReLU(True)
            ))
        self.upsample_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2),
                    nn.ReLU(True)
            ))

        for i in range(3):
            self.mixs.append(Mix(dims[3-i]))

        self.out = nn.Sequential(
            nn.ConvTranspose2d(dims[0], out_chans, kernel_size=5,padding=2),
        )

        self.head = nn.Linear(dims[-1], num_classes)

        # self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()

        patch = 2**3

        mod_pad_h = (patch - h % patch) % patch
        mod_pad_w = (patch - w % patch) % patch
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):

        H, W = x.shape[2:]
        x = self.check_image_size(x)

        i = 1
        arr = []

        x = x * 0.5 + 0.5

        input = x
        
        k = x
        for i in range(4):
            print(k.shape)

            k = self.downsample_layers[i](k)
            k = self.stages[i](k)
            if i != 0:
                arr.append(k)
            
        k = self.mixs[0](k, arr[2])
        k = self.upsample_layers[0](k)
        k = self.mixs[1](k, arr[1])
        k = self.upsample_layers[1](k)
        k = self.mixs[2](k, arr[0])
        k = self.upsample_layers[2](k)

        k = self.out(k)

        output = k * input - k + 1
        output = output * 2 - 1

        output = output[..., :H, :W]
        return output



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_b(pretrained=False, in_22k=False, **kwargs):
    # ------- 12-30 -------
    # 这里第一次设置了64, 128, 256, 512  同时之前的stem 卷积核设置成为了 9  后面的就是卷积核为2 stride为2 PSNR 23.06 SSIM:0.8154  依然没有超过gunet  显存占用  12G
    # ------- 12-31 -------
    # 设置了  32, 32*2, 32*3, 32*4 同时stem卷积核为5 然后本来是想着卷积核都保持和之前一样 kernel_size=2 stride=2 但是特征图输出有问题 于是就把最后一个卷积核设置成了3 stride=2  显存占用了6783MiB 
    # PSNR 23.06   SSIM 0.8142  总体评价: 显存占用低, 效果还不错, 但是还是没有超过gunet, 如果将dim调大一些效果可能会更好
    # 注意: 这里的PSNR与SSIM与12-30日 差不多,但是显存优化了挺多的,这里可以看出dims似乎对结果没有影响吗???
    # ------- 1-1 -------
    # 这里将dim 设置成了 96, 192, 384, 768 看看效果如何  results_best_model -> 23.248 | 0.8227  SSIM效果有进步
    # ------- 1-3 -------
    # 这里打算恢复到32, 32*2, 32*3, 32*4  但是把epoch 改成1000试试看
    # ------- 1-4 -------
    # 将所有的conv2d 改成了gated conv2d   之前的32, 32*4, 32*4, 32*4 在上方65行有bug 所以换成32, 32*4, 32*8, 32*16  效果不错  但是epoch 1000轮    results_best_model -> 24.914 | 0.8398
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[32, 32*4, 32*8, 32*16], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def new_convnext_b(pretrained=False, in_22k=False, **kwargs):
    """
    效果并不好
    """
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[320, 640, 1280, 2560], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == "__main__":
    # c = Conv2d(64,96,2,2)
    # arr = torch.randn(1,64,256,256)
    # print(c(arr).shape)

    c = convnext_b().cuda()
    params = sum(p.numel() for p in c.parameters())
    print(params)

    x = torch.randn(1, 3, 1200, 1600).cuda()
    with torch.no_grad():
        print(c(x).shape)