import os
import sys
import time
import math
import torch.nn.functional as F
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import glob
import cv2
import torch
from torchvision.utils import make_grid, save_image
from shutil import get_terminal_size
import os.path as osp
import torch.nn as nn
from pytorch_msssim import ssim
from torchvision import transforms


import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_last_file(path):
    files = glob.glob(os.path.join(path,'*'))
    if len(files) == 0:
      return ''
    files.sort(key=lambda fn: os.path.getmtime(fn) if not os.path.isdir(fn) else 0)
    return files[-1]


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '.log') #'_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################


def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), bit=8):
    '''
    Converts a torch Tensor into an image Numpy array
    Vis: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    norm = float(2**bit) - 1
    #print('before', tensor[:,:,0].max(), tensor[:,:,0].min(), '\t', tensor[:,:,1].max(), tensor[:,:,1].min(), '\t', tensor[:,:,2].max(), tensor[:,:,2].min())
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    #print('clamp ', tensor[:,:,0].max(), tensor[:,:,0].min(), '\t', tensor[:,:,1].max(), tensor[:,:,1].min(), '\t', tensor[:,:,2].max(), tensor[:,:,2].min())
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.detach().numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
        img_np = (img_np * norm).round()
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x


def single_forward(model, inp):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with torch.no_grad():
        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output


def flipx4_forward(model, inp):
    """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    # normal
    output_f = single_forward(model, inp)

    # flip W
    output = single_forward(model, torch.flip(inp, (-1, )))
    output_f = output_f + torch.flip(output, (-1, ))
    # flip H
    output = single_forward(model, torch.flip(inp, (-2, )))
    output_f = output_f + torch.flip(output, (-2, ))
    # flip both H and W
    output = single_forward(model, torch.flip(inp, (-2, -1)))
    output_f = output_f + torch.flip(output, (-2, -1))

    return output_f / 4


####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# def ssim(img1, img2):

#     # k1 = 0.01,k2 = 0.03,they are used to adjust the weight of Brightness and Contrast
#     C1 = (0.03 * 255)**2
#     C2 = (0.01 * 255)**2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()


# def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Vis images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# Perpectual Loss
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)

# Color Loss
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k
    
def lab2bgr(tensor_lab):
    img_bgr = []
    for i in range(tensor_lab.shape[0]):
        tmp = tensor_lab[i].permute(1,2,0).detach().cpu().numpy()
        tmp = (tmp * 255).astype(np.uint8)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2BGR)
        tmp = np.clip(tmp, 0, 255)
        # cv2.imwrite('snapshots/result/_ori_{}.jpg'.format(i),tmp)
        img_bgr.append(tmp)

    return img_bgr

def lab2rgb(tensor_lab):
    img_rgb = torch.zeros_like(tensor_lab)
    for i in range(tensor_lab.shape[0]):
        tmp = tensor_lab[i].permute(1,2,0).detach().cpu().numpy()
        tmp = (tmp * 255).astype(np.uint8)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB)
        tmp = np.clip(tmp, 0, 255)
        img_rgb[i] = torch.from_numpy(tmp).permute(2,0,1)
        # cv2.imwrite('snapshots/result/_ori_{}.jpg'.format(i),tmp)

    return img_rgb

def get_hist(img_ref):
    tmp = img_ref.detach()
    lab_hist = torch.zeros((tmp.shape[0], 3, 256)) # 64 bins
    for i in range(tmp.shape[0]):
        l_hist = torch.histc((tmp[i][0].view(-1)), bins=256, min=0.0, max=1.0) / (tmp.shape[2] * tmp.shape[3])
        a_hist = torch.histc((tmp[i][1].view(-1)), bins=256, min=0.0, max=1.0) / (tmp.shape[2] * tmp.shape[3])
        b_hist = torch.histc((tmp[i][2].view(-1)), bins=256, min=0.0, max=1.0) / (tmp.shape[2] * tmp.shape[3])
        lab_hist[i] = torch.stack([l_hist, a_hist, b_hist], dim=0)

    return lab_hist

def get_gray(img_ref):
    # tmp = img_ref.clone()
    gray = torch.zeros((img_ref.shape[0], 1, img_ref.shape[2], img_ref.shape[3]))
    for i in range(img_ref.shape[0]):
        tmp = img_ref[i].detach().cpu().permute(1,2,0).numpy()
        tmp = (tmp * 255).astype(np.uint8)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2BGR)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        tmp = (np.asarray(tmp) / 255.0)
        tmp = torch.from_numpy(tmp).float()
        gray[i] = tmp.unsqueeze(0)

    return gray

def get_mean_std(img_hist):   # hist:b,3,256
    hist = img_hist.clone()
    mean = torch.zeros((hist.shape[0], 3))
    std = torch.zeros((hist.shape[0], 3))
    var = torch.zeros((hist.shape[0], 3))
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            mean[i][j] = torch.sum(hist[i][j] / 255 * torch.arange(0, 256).float().cuda())
            var[i][j] = torch.pow(torch.sum(hist[i][j] / 255 * torch.pow(torch.arange(0, 256).float().cuda() - mean[i][j], 2)), 0.5)

    return mean.cuda(), var.cuda()

def get_img_mean_std(img_ref):
    tmp = img_ref.clone()
    mean = torch.mean(tmp*256, [2, 3])
    std = torch.std(tmp*256, [2, 3])

    return mean.cuda(), std.cuda()

def get_inhance_img(img_in,mean_in,std_in,mean_ref,std_ref):
    # img = img_in.detach().cpu().numpy()
    # img = (img * 255).astype(np.uint8)
    img = img_in.clone()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = (img[i][j] - mean_in[i][j]) / torch.sqrt(std_in[i][j] + 1e-5) * torch.sqrt(std_ref[i][j] + 1e-5) + mean_ref[i][j]
    return torch.clamp(img, 0, 1)


def get_tv_loss(img):
    b,c,h,w = img.shape
    img = lab2rgb(img)
    tv_loss = []
    for i in range(len(img)):
        img[i] = (img[i] / 255).transpose(2, 0, 1)
        for j in range(img[i].shape[0]):
            dy = np.abs(img[i][j][:-1, :] - img[i][j][1:, :])
            dx = np.abs(img[i][j][:, :-1] - img[i][j][:, 1:])
            tv_loss.append(np.sum(dx) + np.sum(dy))
    tv_loss = torch.from_numpy(np.asarray(tv_loss)).float().cuda()
    return torch.mean(tv_loss / (h * w))

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps
def torchSSIM(tar_img, prd_img):
    return ssim(tar_img, prd_img, data_range=1.0, size_average=True)

def get_ssim(img1, img2):
    # img01 = lab2rgb(img1)
    # img02 = lab2rgb(img2)
    # img01 = img1.clone() * 255
    # img02 = img2.clone() * 255
    result = 1 - ssim(img1, img2, data_range=1.0, size_average=True)
    # check the lab2rgb result
    # save_image(img02[0] / 255, './test_data/1.jpg')
    return result

# Perpectual Loss
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)

def img_interpolation(ori_, mask_):
    ori = ori_.detach().cpu().numpy()
    mask = mask_.detach().cpu().numpy()
    h, w = mask.shape
    while any(False in row for row in mask):
        for i in range(0, h, 3):
            for j in range(0, w, 3):
                tag = False
                x = 100.0
                if ~mask[i][j]:
                    if i > 0 and j > 0 and mask[i-1][j-1]:
                        lu = abs(ori[i-1][j-1] - ori[i][j])*math.sqrt(2)
                        tag = True
                        if (x - lu)>0.00001:
                            x_i,x_j,x = i-1,j-1,lu
                    if j > 0 and mask[i][j-1]:
                        up = abs(ori[i][j-1] - ori[i][j])
                        tag = True
                        if (x - up)>0.00001:
                            x_i,x_j,x = i,j-1,up
                    if (i+1) < h and j > 0 and mask[i+1][j-1]:
                        ru = abs(ori[i+1][j-1] - ori[i][j])*math.sqrt(2)
                        tag = True
                        if (x - ru)>0.00001:
                            x_i,x_j,x = i+1,j-1,ru
                    if i > 0 and mask[i-1][j]:
                        left = abs(ori[i-1][j] - ori[i][j])
                        tag = True
                        if (x - left)>0.00001:
                            x_i,x_j,x = i-1,j,left
                    if (i+1) < h and mask[i+1][j]:
                        right = abs(ori[i+1][j] - ori[i][j])
                        tag = True
                        if (x - right)>0.00001:
                            x_i,x_j,x = i+1,j,right
                    if i > 0 and (j+1) < w and mask[i-1][j+1]:
                        ld = abs(ori[i-1][j+1] - ori[i][j])*math.sqrt(2)
                        tag = True
                        if (x - ld)>0.00001:
                            x_i,x_j,x = i-1,j+1,ld
                    if (j+1) < w and mask[i][j+1]:
                        down = abs(ori[i][j+1] - ori[i][j])
                        tag = True
                        if (x - down)>0.00001:
                            x_i,x_j,x = i,j+1,down
                    if (i+1) < h and (j+1) < w and mask[i+1][j+1]:
                        rd = abs(ori[i+1][j+1] - ori[i][j])*math.sqrt(2)
                        tag = True
                        if (x - rd)>0.00001:
                            x_i,x_j,x = i+1,j+1,rd
                    if tag:
                        mask[i][j] = True
                        ori[i][j] = ori[x_i][x_j]
                    else:
                        continue
                else:
                    continue

    return ori

def global_transfer(img, bk, mask):
    img_np = img[0].detach().cpu().numpy()*255
    img_lab = cv2.cvtColor(np.transpose(img_np, (1, 2, 0)), cv2.COLOR_RGB2LAB)
    bk = bk[0].detach().cpu().numpy()*255
    bk_lab = cv2.cvtColor(np.transpose(bk, (1, 2, 0)), cv2.COLOR_RGB2LAB)
    mask = np.transpose(mask[0].detach().cpu().numpy(), (1, 2, 0))
    count = np.count_nonzero(mask)
    # 分别计算三个通道的均值和方差
    mean_l, var_l = cv2.meanStdDev(img_lab[:,:,0])
    mean_a, var_a = cv2.meanStdDev(img_lab[:,:,1])
    mean_b, var_b = cv2.meanStdDev(img_lab[:,:,2])

    mean_bk_l, var_bk_l = cv2.meanStdDev(bk_lab[:,:,0])
    mean_bk_a, var_bk_a = cv2.meanStdDev(bk_lab[:,:,1])
    mean_bk_b, var_bk_b = cv2.meanStdDev(bk_lab[:,:,2])

    res_l = (bk_lab[:,:,0] - mean_bk_l)/var_bk_l * var_l + mean_l
    res_a = (bk_lab[:,:,1] - mean_bk_a)/var_bk_a * var_a + mean_a
    res_b = (bk_lab[:,:,2] - mean_bk_b)/var_bk_b * var_b + mean_b
    
    res = np.concatenate((np.expand_dims(res_l, axis=2), np.expand_dims(res_a, axis=2), np.expand_dims(res_b, axis=2)), axis=2)
    # res = img_lab + res * ~mask
    res = cv2.convertScaleAbs(res)
    res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB)
    cv2.imwrite('test_data/ori.png', res)

    return res


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
    
class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
    
def get_mask(seg_result, d):
    masks_out = []
    masks_ins_out = []
    for i in range(len(seg_result)):
        mask_sal_resize = d > 0.08
        # if seg_result[i].masks is not None and len(seg_result[i].masks.data) > 0:
        if seg_result[i] != 0 and seg_result[i].masks is not None:
            mask_ins = seg_result[i].masks.data > 0.5
        else:
            mask_ins = mask_sal_resize[i]
        # calculate the iou
        intersection = mask_sal_resize[i] & mask_ins
        union = mask_sal_resize[i] | mask_ins
        intersection_area = intersection.sum(dim=(1, 2))
        union_area = union.sum(dim=(1, 2))
        iou = intersection_area / union_area

        # test the segment anything
        # count_seg = 0
        # for seg in mask_ins:
        #     seg_ins = seg.unsqueeze(0) * img_ref[i]
        #     seg_ins = transforms.functional.to_pil_image(seg_ins)
        #     seg_ins.save('./snapshots/Sam_seg_instances/seg_{}_{}.png'.format(i, count_seg))
        #     count_seg += 1

        # get the multi object
        index_sal = iou.argmax()
        mask_multi = mask_ins[index_sal]
        for p in range(len(iou)):
            if iou[p] > 0.20:
                mask_multi = mask_multi | mask_ins[p]
            elif iou[p] < 0.05 and iou[p] > 1e-7 and (mask_multi & mask_ins[p]).sum()/mask_ins[p].sum() < 0.7:
            # elif iou[p] < 0.05 and (mask_multi & mask_ins[p]).sum()/mask_ins[p].sum() > 0.5:
                mask_multi = mask_multi & ~mask_ins[p]
            # elif mask_multi.sum() < mask_sal_resize[i].sum():
            #     mask_multi = mask_multi | mask_sal_resize[i]
        if mask_multi.sum() < mask_sal_resize[i].sum():
            mask_multi = mask_multi | mask_sal_resize[i]
        if len(mask_multi.shape) < 3:
            mask_multi = mask_multi.unsqueeze(0)
        masks_ins_out.append(mask_ins[0])
        masks_out.append(mask_multi)
    return masks_out