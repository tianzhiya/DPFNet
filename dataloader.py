import os
import torch
import numpy as np
import torch.utils.data as data
import cv2
import glob

def populate_train_list(images_path):
    vis_list = glob.glob(os.path.abspath(images_path + '/Vis/*'))
    ir_list = glob.glob(os.path.abspath(images_path + '/Ir/*'))

    # ir_list = map(
    #     lambda x: x.replace('Vis', 'Ir'),vis_list)

    return list(vis_list), list(ir_list)

class lowlight_loader(data.Dataset):

    def __init__(self, images_path, height, width):

        self.vis_list, self.ir_list = populate_train_list(images_path)
        self.vis_list.sort()
        self.ir_list.sort()

        self.resize = True
        self.height = height
        self.width = width

        print("Total training examples:", len(self.vis_list))


    def __getitem__(self, index):

        # 读取可见光（RGB） - 默认就是三通道
        vis_input = cv2.imread(self.vis_list[index], cv2.IMREAD_COLOR)

        # 读取红外图（单通道）
        ir_input = cv2.imread(self.ir_list[index], cv2.IMREAD_GRAYSCALE)

        # H >= W，则进行旋转
        if vis_input.shape[0] >= vis_input.shape[1]:
            vis_input = cv2.transpose(vis_input)

        if ir_input.shape[0] >= ir_input.shape[1]:
            ir_input = cv2.transpose(ir_input)

        # resize
        if self.resize:
            vis_input = cv2.resize(vis_input, (self.height, self.width))
            ir_input = cv2.resize(ir_input, (self.height, self.width))

        # 原始图像返回（BGR形式）
        vis_input_ori = vis_input.copy()
        ir_input_ori = ir_input.copy()

        # 可见光：BGR→RGB
        vis_input = cv2.cvtColor(vis_input, cv2.COLOR_BGR2RGB)

        # 归一化 (H, W, C)
        vis_input = vis_input.astype(np.float32) / 255.0
        ir_input = ir_input.astype(np.float32) / 255.0  # 单通道

        # 转Tensor，维度变为 (C, H, W)
        vis_input = torch.from_numpy(vis_input).permute(2, 0, 1)  # 3 x H x W
        ir_input = torch.from_numpy(ir_input).unsqueeze(0)  # 1 x H x W

        return vis_input, ir_input, vis_input_ori, ir_input_ori, self.vis_list[index], self.ir_list[index]

    def __len__(self):
        return len(self.vis_list)