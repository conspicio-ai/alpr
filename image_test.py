import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2
from PIL import Image
import time

from helper import *
from models.enet.model import *

torch.cuda.set_device(0)

alpha = 0.7
# mean = [0.28689554, 0.32513303, 0.28389177]
# std = [0.18696375, 0.19017339, 0.18720214]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = mean, std = std)
            ])

net = ENet(num_classes = 1)
net.load_state_dict(torch.load('saved_models/testing_opt.pt', map_location = 'cpu'))
net.cuda()
net.eval()



frame = cv2.imread('dl/dataset/valid_sih/images/3ecd5af7c4f11963.jpg')

frame = cv2.resize(frame, (640, 480))
frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_pil = Image.fromarray(frame2)
frame_tf = transform(frame_pil)
frame_tf = frame_tf.cuda()
frame_tf = frame_tf.unsqueeze(0)
out = net(frame_tf)

out = torch.sigmoid(out)
out = out.squeeze(1).detach().cpu()
# t = torch.Tensor([0.5])
# out = (out > t)
out = out.numpy()

temp = np.zeros(frame.shape, np.uint8)
temp[:, :, 1] = out[0] * 255

segmented = cv2.addWeighted(frame, alpha, temp, (1 - alpha), 0.0)
cv2.imshow('segmentation_out', segmented)


