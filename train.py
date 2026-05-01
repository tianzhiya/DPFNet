import os
import numpy as np

from GradFusionloss import GradFusionloss
from FusionLoss import FusionLoss
from TaskFusion_dataset import Fusion_dataset
from archs.DPFNet import DPFNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random

random.seed(777)

import argparse

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

import torch
import torch.nn.functional as F


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def gradient_loss(pred, ref):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_ref = ref[:, :, :, 1:] - ref[:, :, :, :-1]
    dy_ref = ref[:, :, 1:, :] - ref[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_ref) + F.l1_loss(dy_pred, dy_ref)


def laplacian_loss(pred, ref):
    lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=pred.device).unsqueeze(
        0).unsqueeze(0)
    lap_pred = F.conv2d(pred, lap_kernel, padding=1)
    lap_ref = F.conv2d(ref, lap_kernel, padding=1)
    return F.l1_loss(lap_pred, lap_ref)


def get_amplitude(img):
    fft = torch.fft.fft2(img, norm="ortho")
    amp = torch.abs(fft)
    return amp


from pytorch_msssim import ssim


def amplitude_max_loss(vis_img,
                       ir_img,
                       vis_amp_out,
                       ir_amp_out,
                       lambda_amp1=1.0,
                       lambda_amp2=0.5):
    vis_amp = get_amplitude(vis_img)
    ir_amp = get_amplitude(ir_img)

    A_max = torch.max(vis_amp, ir_amp)

    loss_l1 = F.l1_loss(vis_amp_out, A_max) + \
              F.l1_loss(ir_amp_out, A_max)

    loss_ssim = (1 - ssim(vis_amp_out, A_max, data_range=1.0, size_average=True)) + \
                (1 - ssim(ir_amp_out, A_max, data_range=1.0, size_average=True))

    loss = lambda_amp1 * loss_l1 + lambda_amp2 * loss_ssim

    return loss


def get_phase(img):
    fft = torch.fft.fft2(img, norm="ortho")
    phase = torch.angle(fft)
    return phase


def phase_max_loss(vis_input_Y,
                   image_ir,
                   vis_img,
                   ir_img,
                   lambda_phase1=1.0,
                   lambda_phase2=1.0,
                   lambda_phase3=1.0,
                   lambda_phase4=1.0):
    pha_vis_out = get_phase(vis_input_Y)
    pha_ir_out = get_phase(image_ir)

    vis_pha = get_phase(vis_img)
    ir_pha = get_phase(ir_img)

    pha_max = torch.max(vis_pha, ir_pha)
    pha_min = torch.min(vis_pha, ir_pha)

    loss_vis_upper = F.relu(pha_vis_out - pha_max)
    loss_vis_lower = F.relu(pha_min - pha_vis_out)

    loss_ir_upper = F.relu(pha_ir_out - pha_max)
    loss_ir_lower = F.relu(pha_min - pha_ir_out)

    loss = (
            lambda_phase1 * loss_vis_upper +
            lambda_phase2 * loss_vis_lower +
            lambda_phase3 * loss_ir_upper +
            lambda_phase4 * loss_ir_lower
    )

    return torch.mean(loss)


def snr_guided_fusion_loss(fuse_img, vis_img, ir_img, snr_mask):
    target = snr_mask * vis_img + (1.0 - snr_mask) * ir_img
    loss_snr = torch.mean(torch.abs(fuse_img - target))
    return loss_snr


def train(config):
    dPFNet = DPFNet().cuda()

    if config.load_pretrain == True:
        dPFNet.load_state_dict(torch.load(config.pretrain_dir))

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    train_dataset = Fusion_dataset('train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)

    fusionLosss = FusionLoss()

    optimizer = torch.optim.Adam(dPFNet.parameters(),
                                 lr=config.lr, weight_decay=config.weight_decay)

    if len(device_ids) > 1:
        dPFNet = nn.DataParallel(dPFNet, device_ids=device_ids)

    print('==> Training start: ')
    for epoch in range(config.num_epochs):
        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            dPFNet.train()
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()
            vis_input_Y = image_vis_ycrcb[:, :1]

            outPuts = dPFNet(
                vis_input_Y, image_ir)
            optimizer.zero_grad()

            loss, loss_dict = fusionLosss(
                outPuts,
                image_ir,
                vis_input_Y,
                outPuts['ir_LL3'],  # 低频
                outPuts['vis_H3']  # 高频 tuple
            )

            loss.backward()
            optimizer.step()
            loss_print = 0
            loss_print = loss_print + loss.item()
            if epoch % 2 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                                                                                            epoch,
                                                                                            len(train_loader),
                                                                                            loss_print,
                                                                                            optimizer.param_groups[0][
                                                                                                'lr']))
                print(
                    f"Total: {loss.item():.4f} | "
                    f"Low: {loss_dict['loss_low'].item():.4f} | "
                    f"High: {loss_dict['loss_high'].item():.4f} | "
                    f"Grad: {loss_dict['loss_grad'].item():.4f} | "
                    f"SSIM: {loss_dict['loss_ssim'].item():.4f} | "
                    f"Freq: {loss_dict['loss_freq'].item():.4f}"
                )

                torch.save(dPFNet.state_dict(), 'checkpoints/DPFNet_{}.pth'.format(epoch))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--lowlight_images_path', type=str, default="./train")
    parser.add_argument('--val_images_path', type=str, default="./val")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--val_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=50)
    parser.add_argument('--snapshot_iter', type=int, default=50)
    parser.add_argument('--image_height', type=int, default=640)
    parser.add_argument('--image_width', type=int, default=480)
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--val_height', type=int, default=400)
    parser.add_argument('--val_width', type=int, default=320)
    config = parser.parse_args()

    for epoch in range(0, 3):
        train(config)
