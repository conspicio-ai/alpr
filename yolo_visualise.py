import cv2,os,numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

img_dir = '/home/himanshu/OIDv4_ToolKit/OID/Dataset/train/Van'
label_dir = '/home/himanshu/sih_number_plate/yolo_label_Van'
files = list(os.listdir(img_dir))

def visualise_yolo_dataset(image_file,label_file):
    with open(label_file, "r") as f:
        label = f.read()

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_h,img_w = img.shape[0], img.shape[1]

    labels = label.strip().split('\n')
    for label in labels:
        label= label.split()
        n_x,n_y = float(label[1]),float(label[2]) 
        n_w,n_h = float(label[3]),float(label[4]) 

        x_c,y_c = int(img_w*n_x) , int(img_h * n_y)
        w,h =  int(img_w*n_w) , int(img_h * n_h)

        x1,y1 = x_c-w//2 , y_c-h//2
        x2,y2 = x_c+w//2 , y_c+h//2

        img = cv2.rectangle(img, (x1,y1),(x2,y2),[0,255,255],3)
    return img


img = visualise_yolo_dataset('/home/himanshu/OIDv4_ToolKit/OID/Dataset/train/Van/Van_0.jpg','/home/himanshu/sih_number_plate/yolo_labels_Van/Van_0.txt')
plt.imshow(img)
plt.show()