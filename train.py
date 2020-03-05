import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from tqdm import tqdm

from helper import *
from models.enet.model import *

os.system('rm -rf /home/himanshu/dl/dataset/valid_sih/images/.ipynb_checkpoints/')
os.system('rm -rf /home/himanshu/dl/dataset/valid_sih/masks/.ipynb_checkpoints/')
torch.cuda.set_device(1)
print('\n', torch.cuda.get_device_name(0),'\n')


num_epochs = 1500
batchSize = 20

mean = [0.28689554, 0.32513303, 0.28389177]
std = [0.18696375, 0.19017339, 0.18720214]
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = mean, std = std)
            ])
trainset = CityscapesDataset(transform = transform, size = 2)
valset = CityscapesDataset(image_path = 'images', transform = transform, size = 2)
trainloader = data.DataLoader(valset, batch_size = batchSize, shuffle = True, drop_last = True)
valloader = data.DataLoader(valset, batch_size = batchSize, shuffle = True, drop_last = True)

net = ENet(num_classes = 1)
# net.load_state_dict(torch.load('saved_models/new_road3.pt', map_location = 'cpu'))
net = net.to('cuda:0')

optimizer = torch.optim.Adam(net.parameters(), lr = 1e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-2, steps_per_epoch = len(trainloader), epochs = num_epochs)
# optimizer.load_state_dict(torch.load('saved_models/new_road3_opt.pt'))
criterion = nn.BCEWithLogitsLoss()
        
j=0
highest_iou = 0
for epoch in range(num_epochs) :
    highest_iou = train(model = net, train_loader = trainloader, val_loader = valloader,loss_function = criterion, optimiser = optimizer, scheduler = scheduler, epoch = epoch, num_epochs = num_epochs, savename = 'saved_models/testing.pt', highest_iou = highest_iou,j=j)
    j+=1
