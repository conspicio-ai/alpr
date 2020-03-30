from __future__ import division
from util import *
from darknet import Darknet
from torch.autograd import Variable
# from preprocess import inp_to_image

import pickle as pkl
import numpy as np

import cv2
import time
import torch




def get_closest(box, labels):
    
    areas = []
    if len(box) == 0 or len(labels) == 0:  ###### This can be changes to return anything you want.
        return None, None

    for x1, y1, x2, y2 in box:
        areas.append((x2-x1)*(y2-y1))

    return box[areas.index(max(areas))], labels[areas.index(max(areas))]


def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img, classes, your_class):
    """
    Draws bounding boxes and writes labels to the image
    """
    
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    if label in your_class:
        color = (0,255,0)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img


def yolo_output(frame, model, your_class, CUDA, inp_dim, coco_names,confidence=0.25, nms_thesh=0.4):
    """
    Get the labeled image and the bounding box coordinates.

    """

    num_classes = 80
    bbox_attrs = 5 + num_classes
    img, orig_im, dim = prep_image(frame, inp_dim)

    im_dim = torch.FloatTensor(dim).repeat(1,2)

    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()

    output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
#            im_dim = im_dim.repeat(output.size(0), 1)
    output[:,[1,3]] *= frame.shape[1]
    output[:,[2,4]] *= frame.shape[0]

    classes = load_classes(coco_names)
    box = []
    labels = []

    #This is where the magic happens
    list(map(lambda x: write(x, orig_im, classes, your_class), output))

    for i in range(output.shape[0]):
        if classes[int(output[i, -1])] in your_class:
            c1 = tuple(output[i,1:3].int())
            c2 = tuple(output[i,3:5].int())
            box.append([c1[0].item(),c1[1].item(), c2[0].item(),c2[1].item()])
            labels.append(classes[int(output[i, -1])])

    return orig_im, box, labels