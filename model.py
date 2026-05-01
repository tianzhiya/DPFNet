import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import arch_util
from SFBlock import AmplitudeNet_skip, SFNet
import functools

import kornia
import torch.nn.functional as F


# 残差块
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


# 残差块
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class PhaseEnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh()  # 限制相位扰动
        )

    def forward(self, pha):
        delta = self.net(pha)
        return pha + 0.1 * delta  # 残差式相位增强


class RegionWareNet(nn.Module):

    def __init__(self):
        super(RegionWareNet, self).__init__()
        # self.r_net = r_net()
        self.relu = nn.ReLU(inplace=True)
        # self.r2_net = r2_net()
        self.fuse_net = FuseNet()
        self.gelu = nn.GELU()
        # self.b_net = b_net()
        self.mFusionNet = FusionNet(output=1)

        # AMPLITUDE ENHANCEMENT
        self.AmpNet = nn.Sequential(
            AmplitudeNet_skip(8),
            nn.Sigmoid()
        )

        self.nf = nf = 64

        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        self.conv_first_1 = nn.Conv2d(4, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, 1)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, 1)

        self.upconv1 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = SFNet(nf)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)

        # PHASE ENHANCEMENT
        self.VisPhaseNet = PhaseEnhanceNet()
        self.IRPhaseNet = PhaseEnhanceNet()

    def get_mask(self, vis, ir):
        fft_vis = torch.fft.fft2(vis)
        fft_ir = torch.fft.fft2(ir)

        mag_vis = torch.abs(fft_vis)
        mag_ir = torch.abs(fft_ir)

        H, W = vis.shape[-2:]
        fy = torch.fft.fftfreq(H, device=vis.device).view(-1, 1)
        fx = torch.fft.fftfreq(W, device=vis.device).view(1, -1)
        freq = torch.sqrt(fx ** 2 + fy ** 2).unsqueeze(0).unsqueeze(0)

        hf = (freq > 0.15).float()

        vis_hf = mag_vis * hf
        ir_hf = mag_ir * hf

        mask = vis_hf / (torch.abs(vis_hf - ir_hf) + 1e-6)
        mask = mask.mean(dim=(2, 3), keepdim=True)
        mask = mask / (mask.max() + 1e-6)

        return mask.repeat(1, 1, H, W).clamp(0, 1)

    def freq_enhance(self, x, phase_net):
        """
        Frequency-domain enhancement: amplitude + phase
        Args:
            x         : [B,1,H,W] 输入图像
            phase_net : nn.Module 相位增强网络
        Returns:
            x_rec    : [B,1,H,W] 重建图像
            mag_enh  : [B,1,H,W] 增强后的振幅
            pha_enh  : [B,1,H,W] 增强后的相位
        """
        # FFT
        fft = torch.fft.fft2(x, norm='backward')
        mag = torch.abs(fft)
        pha = torch.angle(fft)

        # amplitude enhancement（保持原逻辑）
        curve = self.AmpNet(x)
        mag_enh = mag / (curve + 1e-8)

        # phase enhancement（残差轻量增强）
        pha_enh = phase_net(pha)

        # IFFT 重建
        real = mag_enh * torch.cos(pha_enh)
        imag = mag_enh * torch.sin(pha_enh)
        x_rec = torch.fft.ifft2(torch.complex(real, imag),
                                s=x.shape[-2:], norm='backward').real

        return x_rec, mag_enh, pha_enh

    def forward(self, vis_input, ir_input):
        """
        vis_input: [B, 1, H, W]
        ir_input : [B, 1, H, W]
        """

        B, _, H, W = vis_input.shape

        # ====================================================
        # 1. Frequency-domain Amplitude Modulation (IR & VIS)
        # ====================================================

        vis_amp, vis_mag, vis_pha = self.freq_enhance(vis_input, self.VisPhaseNet)
        ir_amp, ir_mag, ir_pha = self.freq_enhance(ir_input, self.IRPhaseNet)

        # ====================================================
        # 2. Padding (for encoder–decoder)
        # ====================================================
        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate

        if pad_h != 0 or pad_w != 0:
            vis_input = F.pad(vis_input, (0, pad_w, 0, pad_h), "reflect")
            ir_input = F.pad(ir_input, (0, pad_w, 0, pad_h), "reflect")
            vis_amp = F.pad(vis_amp, (0, pad_w, 0, pad_h), "reflect")
            ir_amp = F.pad(ir_amp, (0, pad_w, 0, pad_h), "reflect")

        # ====================================================
        # 3. Shallow Feature Encoding
        # ====================================================
        # 输入通道：VIS + IR + VIS_amp + IR_amp
        x_cat = torch.cat([vis_input, ir_input, vis_amp, ir_amp], dim=1)

        L1_fea_1 = self.lrelu(self.conv_first_1(x_cat))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)

        # ====================================================
        # 4. Region-aware Mask (low-light / thermal emphasis)
        # ====================================================
        # 通常用 VIS 构造 mask，更符合夜间增强逻辑
        mask = self.get_mask(vis_input, ir_input)
        mMask = mask
        mask = F.interpolate(mask,
                             size=(fea.shape[2], fea.shape[3]),
                             mode='nearest')
        mask = mask.repeat(1, fea.shape[1], 1, 1)

        # ====================================================
        # 5. Dual-path Feature Fusion
        # ====================================================
        fea_light = self.recon_trunk_light(fea)  # illumination-sensitive
        fea_global = self.transformer(fea)  # structure / context

        fea = fea_global * (1 - mask) + fea_light * mask

        # ====================================================
        # 6. Decoder & Reconstruction
        # ====================================================
        out = self.recon_trunk(fea)

        out = torch.cat([out, L1_fea_3], dim=1)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = torch.cat([out, L1_fea_2], dim=1)
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        out = torch.cat([out, L1_fea_1], dim=1)
        out = self.lrelu(self.HRconv(out))

        fused = self.conv_last(out)

        # residual with VIS (保持自然纹理)
        fused = fused + vis_input
        fused = fused[:, :, :H, :W]

        return fused, mMask, vis_amp, ir_amp,vis_pha,ir_pha


class r_net(nn.Module):
    def __init__(self, depth=[2, 2, 2, 2]):
        super(r_net, self).__init__()

        base_channel = 32

        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Down_scale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            Down_scale(base_channel * 4),
        ])

        self.middle = nn.Sequential(*[RB(base_channel * 8) for _ in range(depth[3])])

        self.Decoder = nn.ModuleList([
            Up_scale(base_channel * 8),
            BasicConv(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            Up_scale(base_channel * 4),
            BasicConv(base_channel * 4, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Up_scale(base_channel * 2),
            BasicConv(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        self.conv_first = BasicConv(3, base_channel // 2, 3, 1)
        self.conv1 = BasicConv(1, base_channel // 4, 3, 1)
        self.conv2 = BasicConv(base_channel // 4, base_channel // 4, 3, 1)
        self.conv3 = BasicConv(base_channel // 4, base_channel // 2, 3, 1)
        self.conv4 = BasicConv(base_channel, base_channel, 3, 1)
        self.conv5 = BasicConv(base_channel, base_channel // 2, 3, 1)
        self.conv6 = BasicConv(base_channel // 2, base_channel // 4, 3, 1)
        self.conv7 = BasicConv(base_channel // 4, base_channel // 4, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 1, 3, 1, 1)
        self.pce = pce()

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:  # layer:1,4,7
                shortcuts.append(x)
        return x, shortcuts

    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:  # layer:1,4,7
                index = len(shortcuts) - (i // 3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)  # match: x1-shortcuts2;x4-shortcuts1;x7-shortcuts0;
            x = self.Decoder[i](x)
        return x

    # feature_scale[0]:(b,32,h,w) feature_scale[1]:(b,64,h/2,w/2) feature_scale[2]:(b,128,h/4,w/4)
    # def forward(self,img_low, lab_feature, feature_scale):
    def forward(self, img_low):
        # x = self.conv_first(img_low)

        x1 = self.conv1(img_low)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat([x1, x2, x3], dim=1)
        # x = self.conv4(x4)

        x5, shortcuts = self.encoder(x4)
        x5 = self.middle(x5)
        shortcuts = self.pce(img_low, shortcuts)  # img_low maybe replace by x
        x6 = self.decoder(x5, shortcuts)

        x7 = self.conv5(x6)
        x8 = self.conv6(x7)
        x9 = self.conv7(x8)
        x_1 = torch.cat([x7, x8, x9], dim=1)
        x_2 = self.conv_last(x_1)
        img_color = (torch.tanh(x_2) + 1) / 2
        return img_color


class r2_net(nn.Module):
    def __init__(self, depth=[2, 2, 2, 2]):
        super(r2_net, self).__init__()

        base_channel = 32

        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            # Down_scale(base_channel),
            BasicConv(base_channel, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Down_scale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            # Down_scale(base_channel*4),
        ])

        self.middle = nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[3])])

        self.Decoder = nn.ModuleList([
            # Up_scale(base_channel*8),
            BasicConv(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            Up_scale(base_channel * 4),
            BasicConv(base_channel * 4, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[1])]),
            # Up_scale(base_channel*2),
            BasicConv(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        self.conv_first = BasicConv(3, base_channel // 2, 3, 1)
        self.conv1 = BasicConv(3, base_channel // 4, 3, 1)
        self.conv2 = BasicConv(base_channel // 4, base_channel // 4, 3, 1)
        self.conv3 = BasicConv(base_channel // 4, base_channel // 2, 3, 1)
        self.conv4 = BasicConv(base_channel, base_channel, 3, 1)
        self.conv5 = BasicConv(base_channel, base_channel // 2, 3, 1)
        self.conv6 = BasicConv(base_channel // 2, base_channel // 4, 3, 1)
        self.conv7 = BasicConv(base_channel // 4, base_channel // 4, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)
        self.pce = pce()

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if i == 0 or i == 3 or i == 5:  # layer:1,4,7 change 0,3,5
                shortcuts.append(x)
        return x, shortcuts

    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if i == 0 or i == 3 or i == 5:  # layer:1,4,7
                if i == 0:
                    index = 2
                elif i == 3:
                    index = 1
                else:
                    index = 0
                # index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)  # match: x1-shortcuts2;x4-shortcuts1;x7-shortcuts0;
            x = self.Decoder[i](x)
        return x

    # feature_scale[0]:(b,32,h,w) feature_scale[1]:(b,64,h/2,w/2) feature_scale[2]:(b,128,h/4,w/4)
    # def forward(self,img_low, lab_feature, feature_scale):
    def forward(self, img_k):
        # x = self.conv_first(img_low)

        x1 = self.conv1(img_k)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat([x1, x2, x3], dim=1)
        # x = self.conv4(x4)

        x5, shortcuts = self.encoder(x4)
        x5 = self.middle(x5)
        # shortcuts = self.pce(img_low, shortcuts)    # img_low maybe replace by d_
        x6 = self.decoder(x5, shortcuts)

        x7 = self.conv5(x6)
        x8 = self.conv6(x7)
        x9 = self.conv7(x8)
        x_1 = torch.cat([x7, x8, x9], dim=1)
        x_2 = self.conv_last(x_1)
        img_color = (torch.tanh(x_2) + 1) / 2
        return img_color


class d2_net(nn.Module):
    def __init__(self, depth=[2, 2, 2, 2]):
        super(d2_net, self).__init__()

        base_channel = 32

        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            # Down_scale(base_channel),
            BasicConv(base_channel, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            # Down_scale(base_channel*2),
            # BasicConv(base_channel*2, base_channel*4, 3, 1),
            # nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            # Down_scale(base_channel*4),
        ])

        self.middle = nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[3])])

        self.Decoder = nn.ModuleList([
            # Up_scale(base_channel*8),
            BasicConv(base_channel * 4, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[2])]),
            # Up_scale(base_channel*4),
            BasicConv(base_channel * 3, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[1])]),
            # Up_scale(base_channel*2),
            # BasicConv(base_channel*2, base_channel, 3, 1),
            # nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        self.conv_first = BasicConv(3, base_channel // 2, 3, 1)
        self.conv1 = BasicConv(base_channel // 2, base_channel, 3, 1)
        self.conv2 = BasicConv(base_channel, base_channel // 2, 3, 1)
        # self.conv3 = BasicConv(base_channel//4, base_channel//2, 3, 1)
        # self.conv4 = BasicConv(base_channel, base_channel, 3, 1)
        # self.conv5 = BasicConv(base_channel, base_channel//2, 3, 1)
        # self.conv6 = BasicConv(base_channel//2, base_channel//4, 3, 1)
        # self.conv7 = BasicConv(base_channel//4, base_channel//4, 3, 1)
        self.conv_last = nn.Conv2d(base_channel // 2, 3, 3, 1, 1)
        self.pce = pce()

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if i % 2 == 0:  # layer:0,2
                shortcuts.append(x)
        return x, shortcuts

    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if i % 2 == 0:  # layer:1,4,7
                index = len(shortcuts) - i // 2 - 1
                x = torch.cat([x, shortcuts[index]], 1)  # match: x1-shortcuts2;x4-shortcuts1;x7-shortcuts0;
            x = self.Decoder[i](x)
        return x

    # feature_scale[0]:(b,32,h,w) feature_scale[1]:(b,64,h/2,w/2) feature_scale[2]:(b,128,h/4,w/4)
    # def forward(self,img_low, lab_feature, feature_scale):
    def forward(self, img_k):
        x = self.conv_first(img_k)

        x1 = self.conv1(x)
        # x2 = self.conv2(x1)
        # x3 = self.conv3(x2)
        # x4 = torch.cat([x1,x2,x3], dim=1)
        # x = self.conv4(x4)

        x2, shortcuts = self.encoder(x1)
        x3 = self.middle(x2)
        # shortcuts = self.pce(img_low, shortcuts)    # img_low maybe replace by d_
        x4 = self.decoder(x3, shortcuts)

        # x7 = self.conv5(x6)
        # x8 = self.conv6(x7)
        x5 = self.conv2(x4)
        # x_1 = torch.cat([x7,x8,x9], dim=1)
        x7 = self.conv_last(x5)
        img_color = (torch.tanh(x7) + 1) / 2
        return img_color


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True,
                 transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
            )
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class RB(nn.Module):  # residual block
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x


class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel * 2, 3, 2)

    def forward(self, x):
        return self.main(x)


class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel // 2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)


class pce(nn.Module):
    # parmid color embedding

    def __init__(self):
        super(pce, self).__init__()

        self.cma_3 = cma(1, 32)
        self.cma_2 = cma(32, 64)
        self.cma_1 = cma(64, 128)

    def forward(self, img, shortcuts):
        # img:(b,3,386,512) 
        # 不同size与channel的灰度图像信息，shortcuts[0]:(1,32,400,600);shortcuts[1]:(1,64,200,300);shortcuts[2]:(1,128,100,150)；
        # change channels
        x_3_color, img_i = self.cma_3(img, shortcuts[0])  # x_3_color利用l1和cos距离计算得到的相似度矩阵；c_2为c上采样得到的结果
        x_2_color, img_i = self.cma_2(img_i, shortcuts[1])
        x_1_color, _ = self.cma_1(img_i, shortcuts[2])

        return [x_3_color, x_2_color, x_1_color]  # 得到不同size的颜色嵌入矩阵


class cma(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(cma, self).__init__()
        # self.conv:Conv2d,InstanceNorm2d,LeakyReLU,Upsample
        self.weight = nn.Parameter(torch.ones(1))
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.InstanceNorm2d(out_channels),
                                  nn.LeakyReLU(0.2, inplace=True))

    def forward(self, c, x):
        # c: img_low
        # x: image features

        c = self.conv(c)
        # l1 distance
        channels = c.shape[1]
        sim_mat_l1 = -torch.abs(x - c)  # <0  (b,c,h,w)=(1,128,100,150)
        sim_mat_l1 = torch.sum(sim_mat_l1, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_l1 = torch.sigmoid(sim_mat_l1)  # (0, 0.5) (b,1,h,w)
        sim_mat_l1 = sim_mat_l1.repeat(1, channels, 1, 1)  # (b,c,h,w)=(1,128,100,150)
        sim_mat_l1 = 2 * sim_mat_l1  # (0, 1)

        # inner product
        sim_mat_cos = x * c  # >0 (b,c,h,w)
        sim_mat_cos = torch.sum(sim_mat_cos, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_cos = torch.tanh(sim_mat_cos)  # (0, 1) (b,1,h,w)
        sim_mat_cos = sim_mat_cos.repeat(1, channels, 1, 1)  # (0, 1)

        # similarity matrix
        sim_mat = sim_mat_l1 * sim_mat_cos  # (0, 1)

        # color embeding
        x_color = x + c * sim_mat * self.weight  # (b,c,h,w)=(1,128,100,150)

        # color features upsample
        c_down = F.interpolate(c, scale_factor=0.5, mode='bilinear', align_corners=True)  # (b,c,h,w)=(1,128,200,300)

        return x_color, c_down


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]  # 2
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]  # 2
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


##########################################################################
class FuseNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(FuseNet, self).__init__()

        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
                                    num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x3_img):
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img = x3_img[:, :, 0:int(H / 2), :]
        x2bot_img = x3_img[:, :, int(H / 2):H, :]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x2top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        ## Concat deep features
        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3 = self.shallow_feat3(x3_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))

        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)

        return [stage3_img + x3_img, stage2_img, stage1_img]


class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x


class RGBD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGBD, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        return F.leaky_relu(x1 + x2, negative_slope=0.1)


class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()
        vis_ch = [16, 32, 48]
        inf_ch = [16, 32, 48]
        output = 1
        self.vis_conv = ConvLeakyRelu2d(1, vis_ch[0])
        self.vis_rgbd1 = RGBD(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = RGBD(vis_ch[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv = ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = RGBD(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = RGBD(inf_ch[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2] + inf_ch[2], vis_ch[1] + vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1] + inf_ch[1], vis_ch[0] + inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0] + inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)

    def forward(self, image_vis, image_ir):
        # split data into RGB and INF
        # x_vis_origin = image_vis[:,:1]
        x_vis_origin = image_vis
        x_inf_origin = image_ir
        # encode
        x_vis_p = self.vis_conv(x_vis_origin)
        x_vis_p1 = self.vis_rgbd1(x_vis_p)
        x_vis_p2 = self.vis_rgbd2(x_vis_p1)

        x_inf_p = self.inf_conv(x_inf_origin)

        x_inf_p1 = self.inf_rgbd1(x_inf_p)
        x_inf_p2 = self.inf_rgbd2(x_inf_p1)

        x = self.decode4(torch.cat((x_vis_p2, x_inf_p2), dim=1))
        # x=self.decode4(x)
        x = self.decode3(x)
        x = self.decode2(x)
        x = self.decode1(x)

        return x
