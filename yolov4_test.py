import sys
sys.path.append('pytorch-YOLOv4-master')
from tool.darknet2pytorch import Darknet as DarknetYolov4
import argparse
import cv2,time

from tool.utils import *
from tool.torch_utils import *

use_cuda = True

cfg_v4 = 'pytorch-YOLOv4-master/cfg/yolo-obj.cfg'
name_v4 = 'pytorch-YOLOv4-master/data/num.names'
weight_v4 = 'num_plate.weights'

m = DarknetYolov4(cfg_v4)
m.load_weights(weight_v4)
print('Loading weights from %s... Done!' % (weight_v4))

if use_cuda:
    m.cuda()

cap = cv2.VideoCapture('1.mp4')
# cap = cv2.VideoCapture("./test.mp4")
cap.set(3, 1280)
cap.set(4, 720)
print("Starting the YOLO loop...")

num_classes = m.num_classes

class_names = ['plate']

while True:
    ret, img = cap.read()
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    start = time.time()
    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
    finish = time.time()
    print('Predicted in %f seconds.' % (finish - start))

    result_img = plot_boxes_cv2(img, boxes[0], savename=False, class_names=class_names)
    cv2.imshow('Yolo demo', result_img)
    cv2.waitKey(1)

cap.release()