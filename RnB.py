import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

import torch


class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)
    
smth_3 = GaussianSmoothing(sigma=3.0).cuda()


sobel_x = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32).cuda()

sobel_y = torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=torch.float32).cuda()

sobel_x = sobel_x.view(1, 1, 3, 3)
sobel_y = sobel_y.view(1, 1, 3, 3)

sobel_conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
sobel_conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)


sobel_conv_x.weight = nn.Parameter(sobel_x)
sobel_conv_y.weight = nn.Parameter(sobel_y)

def edge_loss(attn_map, mask, iou):
    
    loss_ = 0
    
    mask_clone = mask.clone()[1:-1, 1:-1]
    
    attn_map_clone = attn_map.unsqueeze(0).unsqueeze(0)
    attn_map_clone = attn_map_clone / attn_map_clone.max().detach()
    attn_map_clone = F.pad(attn_map_clone, (1, 1, 1, 1), mode='reflect')
    attn_map_clone = smth_3(attn_map_clone)

    sobel_output_x = sobel_conv_x(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_output_y = sobel_conv_y(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_sum = torch.sqrt(sobel_output_y ** 2  + sobel_output_x ** 2)
    sobel_sum = sobel_sum 
    
    loss_ += 1 - (sobel_sum * mask_clone).sum() / sobel_sum.sum() * (1 - iou)
    
    return loss_


def compute_rnb(attn_maps_mid, attn_maps_up, attn_maps_down, attn_self, bboxes, object_positions, iter=None, attn_weight=None):

    
    
    loss = 0
    object_number = len(bboxes)

    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()

    attn16_list = []
    for attn_map_integrated in attn_maps_up[0]:
        attn16_list.append(attn_map_integrated)
        
    for attn_map_integrated in attn_maps_down[-1]:
        attn16_list.append(attn_map_integrated)
    
    attn_all_list = []
    attn_edge = []
    for sub_list in attn_maps_up:
        for item in sub_list:
            b, i, j = item.shape
            sub_res = int(math.sqrt(i))
            item = item.reshape(b, sub_res, sub_res, j).permute(3, 0, 1, 2).mean(dim=1, keepdim=True)
            if sub_res <= 32:
                attn_all_list.append(F.interpolate(item, 64, mode='bilinear'))
                attn_edge.append(F.interpolate(item, 64, mode='bilinear'))

    
    for sub_list in attn_maps_down:
        for item in sub_list:
            b, i, j = item.shape
            sub_res = int(math.sqrt(i))
            item = item.reshape(b, sub_res, sub_res, j).permute(3, 0, 1, 2).mean(dim=1, keepdim=True)
            if sub_res < 32:
                attn_all_list.append(F.interpolate(item, 64, mode='bilinear'))
            
    
    
    for item in attn_maps_mid:
        b, i, j = item.shape
        sub_res = int(math.sqrt(i))
        item = item.reshape(b, sub_res, sub_res, j).permute(3, 0, 1, 2).mean(dim=1, keepdim=True)
        attn_all_list.append(F.interpolate(item, 64, mode='bilinear'))
        attn_edge.append(F.interpolate(item, 64, mode='bilinear'))
        
    
    attn_all_list = torch.cat(attn_all_list, dim=1)
    attn_all_list = attn_all_list.mean(dim=1).permute(1,2,0)
    attn_all = attn_all_list[:, :, 1:]
    
    attn_edge = torch.cat(attn_edge, dim=1)
    attn_edge = attn_edge.mean(dim=1).permute(1,2,0)
    attn_edge = torch.nn.functional.softmax(attn_edge[:, :, 1:]*120, dim=-1)
    
    
    H= W = 64

    obj_loss = 0
    
    rows, cols = torch.meshgrid(torch.arange(H), torch.arange(W))
    positions = torch.stack([rows.flatten(), cols.flatten()], dim=-1)
    positions = positions.to(attn_all.device) / H
    
    # import ipdb; ipdb.set_trace()
    for obj_idx in range(object_number):

        for num, obj_position in enumerate(object_positions[obj_idx]):
            true_obj_position = obj_position - 1
            # print(obj_position)
            if num == 0:
                att_map_obj_raw = attn_all[:, :, true_obj_position]
                att_map_edge = attn_edge[:, :, true_obj_position]

            else:
                att_map_obj_raw = att_map_obj_raw + attn_all[:, :, true_obj_position]
                att_map_edge = att_map_edge + attn_edge[:, :, true_obj_position]
        
        attn_norm = (att_map_obj_raw - att_map_obj_raw.min()) / (att_map_obj_raw.max() - att_map_obj_raw.min())


        mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
        mask_clone = mask.clone()
        
        for obj_box in bboxes[obj_idx]:
            x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
            int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
            mask[y_min: y_max, x_min: x_max] = 1
        
        mask_none_cls = (1 - mask)
            
        threshold = (attn_norm * mask).sum() / mask.sum() / 5 * 2 + \
            ((attn_norm * mask_none_cls).sum() / mask_none_cls.sum() / 5 * 3) if mask_none_cls.sum() != 0 else 0
    
        thres_image = attn_norm.gt(threshold) * 1.0
        noise_image = F.sigmoid(20 * (attn_norm - threshold))
        
        rows, cols = torch.where(thres_image > 0.3)
        x1, y1 = cols.min(), rows.min()
        x2, y2 = cols.max(), rows.max()
        
        mask_aug = mask_clone
        mask_aug[y1: y2, x1: x2] = 1    
        mask_aug_in = mask_aug * mask 
        iou = (mask_aug * mask).sum() / torch.max(mask_aug, mask).sum()
            
        if iou < 0.85:
            
            this_cls_diff_aug_1 = (mask_aug - attn_norm).detach() + attn_norm
            this_cls_diff_aug_in_1 = (mask_aug_in - attn_norm).detach() + attn_norm
            
            obj_loss += 1 - (1 - iou) * (mask * this_cls_diff_aug_in_1).sum() * (1 / this_cls_diff_aug_1.sum().detach())
            obj_loss += 1 - (1 - iou) * (mask * this_cls_diff_aug_in_1).sum().detach() * (1 / this_cls_diff_aug_1.sum())
            
            if object_number > 1 and obj_idx > -1:

                if (att_map_obj_raw * mask).max() < (att_map_obj_raw * (1 - mask)).max():
                    obj_loss += edge_loss(att_map_edge, mask, iou) * 1 
    

            obj_loss += 1 - (1 - iou) * ((mask * noise_image).sum() * (1 / noise_image.sum().detach())) * 0.5
            obj_loss += 1 - (1 - iou) * ((mask * noise_image).sum().detach() * (1 / noise_image.sum())) * 0.5
                
            
    loss += obj_loss / object_number

    return loss, attn_weight

