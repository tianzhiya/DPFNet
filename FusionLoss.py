import torch
import torch.nn as nn
import torch.nn.functional as F

from GradFusionloss import GradFusionloss


# =========================
# 梯度计算（Sobel）
# =========================
def gradient(x):
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)

    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return grad


# =========================
# SSIM（简化版）
# =========================
def ssim_loss(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return 1 - ssim_map.mean()


# =========================
# 总损失函数
# =========================
class FusionLoss(nn.Module):
    def __init__(self,
                 lambda_low=1.0,
                 lambda_high=1.0,
                 lambda_grad=50000,
                 lambda_ssim=0.5,
                 lambda_freq=0.2):
        super(FusionLoss, self).__init__()

        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.lambda_grad = lambda_grad
        self.lambda_ssim = lambda_ssim
        self.lambda_freq = lambda_freq
        self.fusionLosss = GradFusionloss()

    def forward(self, outputs, ir_img, vis_img,
                ir_LL, vis_H):
        """
        outputs:
            dict:
                LL_fused: [B,1,H,W]
                H_fused: tuple(3)
                fused_image: [B,1,H,W]

        ir_img: 原始红外
        vis_img: 原始可见光（Y通道）
        ir_LL: 红外低频
        vis_H: 可见光高频 tuple
        """

        LL_fused = outputs['LL_fused']
        H_fused = outputs['H_fused']
        fused = outputs['fused_image']

        # =========================
        # 1. 低频损失（红外主导 + 梯度加权）
        # =========================
        grad_ir = gradient(ir_LL)
        weight = 1 + grad_ir

        loss_low = torch.mean(weight * torch.abs(LL_fused - ir_LL))

        # =========================
        # 2. 高频损失（可见光主导）
        # =========================
        loss_high = 0
        for hf, vh in zip(H_fused, vis_H):
            loss_high += F.l1_loss(hf, vh)

        # =========================
        # 3. 梯度损失（结构保持）
        loss_grad = self.fusionLosss(
            vis_img, ir_img, fused
        )
        loss_grad = loss_grad[0]

        # =========================
        # 4. SSIM损失（视觉质量）
        # =========================
        loss_ssim = ssim_loss(fused, ir_img) + ssim_loss(fused, vis_img)

        # =========================
        # 5. 频域损失（幅值约束）
        # =========================
        fft_fused = torch.fft.rfft2(fused)
        fft_ir = torch.fft.rfft2(ir_img)
        fft_vis = torch.fft.rfft2(vis_img)

        amp_fused = torch.abs(fft_fused)
        amp_ir = torch.abs(fft_ir)
        amp_vis = torch.abs(fft_vis)

        loss_freq = F.l1_loss(amp_fused, amp_ir) + \
                    F.l1_loss(amp_fused, amp_vis)

        # =========================
        # 总损失
        # =========================
        loss_total = (self.lambda_low * loss_low +
                      self.lambda_high * loss_high +
                      self.lambda_grad * loss_grad +
                      self.lambda_ssim * loss_ssim +
                      self.lambda_freq * loss_freq)

        return loss_total, {
            'loss_low': loss_low,
            'loss_high': loss_high,
            'loss_grad': loss_grad,
            'loss_ssim': loss_ssim,
            'loss_freq': loss_freq
        }
