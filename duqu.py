import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
# import matplotlib.pyplot as plt
import torch.nn.functional as f
import torchvision

# toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
# img = cv2.imread('load/Set5/LR/x4/baby.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #将BGR格式转换成RGB格式
# print(img.shape)  # numpy数组格式为（H,W,C）
#
# img_tensor = transforms.ToTensor()(img)  # 把np转换成tensor型，维度也自然转换，
# print(img_tensor.size())
# torchvision.utils.save_image(img_tensor, "out_cv.png")
x = torch.arange(0, 2*4*4).float().view(1,2,4,4)
# print('*x.shape[-2:]', *x.shape[-2:])
# print('x.shape[-2:]', x.shape[-2:])
# x = x.expand(x.shape[0], 2, *x.shape[-2:])
# print(x)
# x1 = f.unfold(x, kernel_size=3, padding=1).view(
#                 x.shape[0], x.shape[1] * 9, x.shape[2], x.shape[3])
# print(x1.shape)
# print(x.shape)
# B, C_kh_kw, L = x1.size()
# x1 = x1.permute(0, 2, 1)
# x1 = x1.view(B, L, -1, 3, 3)
# print(x1)
x1=torch.split(x, 2, dim=-1)
print(x1)
q_freq = torch.stack(x1, dim=-1)
print(q_freq.shape)

