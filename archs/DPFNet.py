from archs import arch_util
from archs.arch_util import Depth_conv
from archs.model_utils import *
from archs.wavelet_normlize import *

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class InnovativeHighFusionBlock(nn.Module):
    """
    改进版：可见光引导高频融合（支持小波三子带）
    修改要点：
    1. 可见光三子带分别建模
    2. 多尺度增强加入激活
    3. Fourier 幅值引导改为三子带共同建模，而不是只用 LH
    4. 注意力融合保持“可见光主导、红外补充”
    """

    def __init__(self, nf=16):
        super(InnovativeHighFusionBlock, self).__init__()
        self.nf = nf

        # 可见光三子带分别建模
        self.conv_LH = nn.Conv2d(1, nf, 3, padding=1)
        self.conv_HL = nn.Conv2d(1, nf, 3, padding=1)
        self.conv_HH = nn.Conv2d(1, nf, 3, padding=1)

        # 红外高频特征提取（3 个子带拼接后输入）
        self.ir_conv = nn.Conv2d(3, nf, 3, padding=1)

        # 多尺度纹理增强
        self.ms_conv1 = nn.Conv2d(nf, nf, 3, padding=1)
        self.ms_conv2 = nn.Conv2d(nf, nf, 5, padding=2)

        # Fourier 幅值增强
        # 这里输入仍然保持 1 通道，因为我们对三子带幅值取均值
        self.fft_conv = nn.Conv2d(1, nf, 1)

        # 注意力门控
        self.attn = nn.Sequential(
            nn.Conv2d(nf * 3, nf, 1),
            nn.Sigmoid()
        )

        # 输出 3 通道，对应融合后的 LH/HL/HH
        self.conv_out = nn.Conv2d(nf, 3, 3, padding=1)

    def _fft_amplitude(self, x):
        """
        对单个子带做 FFT 幅值提取，并 resize 回原始空间尺寸
        x: [B,1,H,W]
        return: [B,1,H,W]
        """
        fft = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fft)
        mag = F.interpolate(
            mag,
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        return mag

    def forward(self, vis_H, ir_H, gradient=None):
        # =========================
        # Step 1: 处理 tuple 高频
        # =========================
        if not isinstance(vis_H, tuple) or len(vis_H) != 3:
            raise ValueError("vis_H must be tuple (LH, HL, HH)")

        LH, HL, HH = vis_H

        if isinstance(ir_H, tuple):
            if len(ir_H) != 3:
                raise ValueError("ir_H must be tuple (LH, HL, HH)")
            ir_H = torch.cat(ir_H, dim=1)   # [B,3,H,W]
        elif not torch.is_tensor(ir_H):
            raise ValueError("ir_H must be tuple or tensor")

        # =========================
        # Step 2: 可见光三子带建模
        # =========================
        feat_LH = F.relu(self.conv_LH(LH), inplace=True)
        feat_HL = F.relu(self.conv_HL(HL), inplace=True)
        feat_HH = F.relu(self.conv_HH(HH), inplace=True)

        # 三方向高频融合
        vis_feat = feat_LH + feat_HL + feat_HH

        # =========================
        # Step 3: 多尺度增强
        # =========================
        ms_feat_3 = F.relu(self.ms_conv1(vis_feat), inplace=True)
        ms_feat_5 = F.relu(self.ms_conv2(vis_feat), inplace=True)
        vis_feat = vis_feat + ms_feat_3 + ms_feat_5

        # =========================
        # Step 4: 红外高频特征
        # =========================
        ir_feat = F.relu(self.ir_conv(ir_H), inplace=True)

        # =========================
        # Step 5: Fourier 幅值引导
        # 改为 LH / HL / HH 三子带共同建模
        # =========================
        mag_LH = self._fft_amplitude(LH)
        mag_HL = self._fft_amplitude(HL)
        mag_HH = self._fft_amplitude(HH)

        # 取均值，保持 1 通道输入 fft_conv
        mag = (mag_LH + mag_HL + mag_HH) / 3.0
        fft_feat = F.relu(self.fft_conv(mag), inplace=True)

        # =========================
        # Step 6: 注意力融合（可见光主导）
        # =========================
        combined = torch.cat([vis_feat, ir_feat, fft_feat], dim=1)
        attn = self.attn(combined)

        H_fused = attn * vis_feat + (1.0 - attn) * ir_feat

        # =========================
        # Step 7: 输出
        # =========================
        H_fused = self.conv_out(H_fused)

        # 分成三个子带输出
        H_fused_tuple = torch.chunk(H_fused, 3, dim=1)

        return H_fused_tuple


class LowFusionBlock(nn.Module):
    """
    红外引导低频融合模块
    输入：
        ir_LL3_gamma: 红外低频分量（经过 gamma 校正） [B,1,H,W]
        vis_LL3: 可见光低频分量 [B,1,H,W]
        gradient: 红外低频梯度 [B,1,H,W]
    输出：
        LL_fused: 融合后的低频分量 [B,1,H,W]
    """

    def __init__(self, nf=16, numblocks=6, in_channels=1):
        super(LowFusionBlock, self).__init__()
        self.nf = nf
        self.dfgf = DFGFLow(nf=nf, numblocks=numblocks, in_channels=in_channels)
        # 可见光特征映射卷积
        self.conv_vis = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1, bias=True)
        # 输出卷积，把 nf 通道映射回 1
        self.conv_out = nn.Conv2d(nf, in_channels, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, ir_LL3_gamma, vis_LL3, gradient):
        # 1. 映射可见光低频特征
        # vis_feat = self.conv_vis(vis_LL3)
        # 2. 红外引导 Fourier 融合
        fused_feat = self.dfgf(vis_LL3, gradient, ir_LL3_gamma)
        # 3. 输出融合低频
        LL_fused = self.conv_out(fused_feat)
        return LL_fused


class DPFNet(nn.Module):
    def __init__(self, y_nf=16, f_nf=16, s_nf=32):
        super(DPFNet, self).__init__()

        # Model parameters
        self.y_nf = y_nf
        self.f_nf = f_nf
        self.s_nf = s_nf

        # Fourier stage processing
        self.fourier_process1 = DFGFLow(nf=self.s_nf, numblocks=2)
        self.fourier_process2 = DFGFLow(nf=self.s_nf, numblocks=4)
        self.fourier_process3 = DFGFLow(nf=self.s_nf, numblocks=6)

        # Down-sampling and up-sampling groups
        self.down_group1 = DownFRG()
        self.down_group2 = DownFRG()
        self.down_group3 = DownFRG()

        self.up_group3 = UpFRG()
        self.up_group2 = UpFRG()
        self.up_group1 = UpFRG()

        # Wavelet transform
        self.dwt = arch_util.DWT()  # Discrete Wavelet Transform

        # High-frequency processing
        self.high_process3 = DFGFHigh(in_nf=3, out_if=3, nf=self.s_nf)
        self.high_process2 = DFGFHigh(in_nf=3, out_if=3, nf=self.s_nf)
        self.high_process1 = DFGFHigh(in_nf=3, out_if=3, nf=self.s_nf)

        # Post-processing with attention
        self.postprocess3 = WTConvAttentionModel(nf=self.s_nf)
        self.postprocess2 = WTConvAttentionModel(nf=self.s_nf)
        self.postprocess1 = SpatialAugmentModel(nf=self.s_nf)

        # Gamma correction network
        self.gammanet = GammaNet(input_channels=1, feature_channels=self.s_nf)

        # Activation function
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.low_fusion_block = LowFusionBlock(nf=16, numblocks=6, in_channels=1)
        self.high_fusion_block = InnovativeHighFusionBlock(nf=16)

    def pad_to_multiple(self, x, multiple=8):
        _, _, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def forward(self, ir_img, vis_Y):
        """
            ir_img: 红外图像 [B, 1, H, W]
            vis_Y: 可见光图像的 Y 通道 [B, 1, H, W]
            """
        outputs = {}

        # ===============================
        # Step 1: 小波/下采样分解
        # ===============================
        # 红外低频/高频
        ir_LL1, ir_H1 = self.down_group1(ir_img)
        ir_LL2, ir_H2 = self.down_group2(ir_LL1)
        ir_LL3, ir_H3 = self.down_group3(ir_LL2)

        # 可见光低频/高频
        vis_LL1, vis_H1 = self.down_group1(vis_Y)
        vis_LL2, vis_H2 = self.down_group2(vis_LL1)
        vis_LL3, vis_H3 = self.down_group3(vis_LL2)

        outputs['ir_LL3'] = ir_LL3
        outputs['vis_H3'] = vis_H3
        # ===============================
        # Step 2: 红外引导低频分支（显著目标）
        # ===============================
        gamma = self.gammanet(ir_LL3)
        ir_LL3_gamma = torch.pow(ir_LL3, gamma)
        gradient3 = arch_util.compute_gradient(ir_LL3_gamma)

        # 红外引导融合低频
        LL_fused = self.low_fusion_block(ir_LL3_gamma, vis_LL3, gradient3)
        outputs['LL_fused'] = LL_fused

        # ===============================
        # Step 3: 可见光引导高频分支（细节纹理）
        # ===============================
        H_fused = self.high_fusion_block(vis_H3, ir_H3, gradient3)
        outputs['H_fused'] = H_fused

        # ===============================
        # Step 4: IDWT / 上采样重建融合图像
        # ===============================
        # ===============================

        # ---------- Level 3 ----------
        # 高频处理
        if isinstance(H_fused, torch.Tensor):
            H3_tuple = torch.chunk(H_fused, 3, dim=1)
        else:
            H3_tuple = H_fused

        x_l2 = self.up_group3(LL_fused, H3_tuple)

        # ---------- Level 2 ----------
        # 使用原始中层高频（可升级成融合版）
        if isinstance(ir_H2, tuple):
            H2 = ir_H2
        else:
            H2 = torch.chunk(ir_H2, 3, dim=1)

        x_l1 = self.up_group2(x_l2, H2)

        # ---------- Level 1 ----------
        if isinstance(ir_H1, tuple):
            H1 = ir_H1
        else:
            H1 = torch.chunk(ir_H1, 3, dim=1)

        fused_image = self.up_group1(x_l1, H1)

        # ---------- 后处理 ----------
        fused_image = self.postprocess3(fused_image)

        outputs['fused_image'] = fused_image

        return outputs


class GammaNet(nn.Module):
    def __init__(self, input_channels=1, feature_channels=16):
        super(GammaNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.gamma_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels * 2, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        gamma = self.gamma_pred(features)
        gamma = gamma.view(-1, 1, 1, 1)
        return gamma



class DFGFLow(nn.Module):
    """
    红外引导低频融合核心模块
    输入：
        x: 可见光低频特征 [B, 1, H, W]
        gradient: 红外低频梯度 [B, 1, H, W]
        x_light: 红外低频 gamma 校正 [B, 1, H, W]
    输出：
        x_amplitude: 融合后的低频特征
    """

    def __init__(self, nf=16, numblocks=6, in_channels=1):
        super(DFGFLow, self).__init__()
        self.s_nf = nf
        self.processblock = LowFrequencyProcessing(nf=self.s_nf, num_blocks=numblocks, input_channels=nf)
        # Initial convolution layers for Fourier features
        self.conv_first_fr = nn.Conv2d(in_channels, self.s_nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_first_map = nn.Conv2d(in_channels, self.s_nf, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, vis_feat, gradient, ir_LL3_gamma):
        # 红外低频引导
        ir_light_fre = torch.fft.rfft2(ir_LL3_gamma, norm='backward')
        ir_light_mag = torch.abs(ir_light_fre)
        ir_light_pha = torch.angle(ir_light_fre)

        # 卷积映射到 nf 通道
        ir_light_mag = self.conv_first_fr(ir_light_mag)
        ir_light_pha = self.conv_first_map(ir_light_pha)

        # 幅值增强
        x_amplitude = self.processblock(vis_feat, ir_light_pha, ir_light_mag)
        return x_amplitude


class DFGFHigh(nn.Module):
    def __init__(self, in_nf=3, out_if=3, nf=16):
        super(DFGFHigh, self).__init__()

        self.conv1_first = nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2_first = nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, groups=4),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, groups=4),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, groups=4),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
        )

        self.convf1_first = nn.Sequential(
            nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            Depth_conv(in_ch=nf, out_ch=nf)
        )
        self.convf2_first = nn.Sequential(
            nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            Depth_conv(in_ch=nf, out_ch=nf)
        )
        self.convf3_first = nn.Sequential(
            nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            Depth_conv(in_ch=nf, out_ch=nf)
        )

        self.convf1 = nn.Sequential(
            Depth_conv(in_ch=nf, out_ch=nf),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
            nn.Conv2d(nf, nf, 3, 1, 1, groups=4)
        )
        self.convf2 = nn.Sequential(
            Depth_conv(in_ch=nf, out_ch=nf),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
            nn.Conv2d(nf, nf, 3, 1, 1, groups=4)
        )

        self.convf2 = nn.Sequential(
            Depth_conv(in_ch=nf, out_ch=nf),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
            nn.Conv2d(nf, nf, 3, 1, 1, groups=4)
        )

        self.conv1_out = nn.Conv2d(nf, out_if, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2_out = nn.Conv2d(nf, out_if, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3_out = nn.Conv2d(nf, out_if, kernel_size=1, stride=1, padding=0, bias=True)

        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, f, fg):
        f1, f2, f3 = f
        f_add = f1 + f2 + f3
        fg = self.conv1_first(fg)
        f_add = self.conv2_first(f_add)
        attention1 = self.sigm(self.conv1(fg))
        attention2 = self.sigm(self.conv2(f_add))
        attention = fg * attention1 + f_add * attention2
        attention = self.sigm(self.conv3(attention))

        f1 = self.convf1_first(f1)
        f1 = f1 + attention * f1
        f1 = self.conv1_out(self.convf1(f1))

        f2 = self.convf1_first(f2)
        f2 = f2 + attention * f2
        f2 = self.conv2_out(self.convf2(f2))

        f3 = self.convf1_first(f3)
        f3 = f3 + attention * f3
        f3 = self.conv3_out(self.convf1(f3))

        return (f1, f2, f3)
