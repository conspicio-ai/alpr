import cv2,os
from tqdm import tqdm
import matplotlib.pyplot as plt

img_dir = '/home/himanshu/Downloads/train/Vehicle_registration_plate'
label_dir = '/home/himanshu/Downloads/train/Label'

save_folder = 'yolo_labels_plate'
# img_dir = '/content/train/Vehicle registration plate'

if not os.path.exists(save_folder):
	os.mkdir(save_folder)

for i, filename in tqdm(enumerate(os.listdir(img_dir))):

	rand_img_path = os.path.join(img_dir,filename)
	abs_label_dir = os.path.join(label_dir,filename[:-3]+'txt')

	img = cv2.imread(rand_img_path)
	img_h,img_w = img.shape[0], img.shape[1]

	with open(abs_label_dir, "r") as f:
		label = f.read()

	label = label.strip().split()

	x_top,y_top = int(float(label[3])),int(float(label[4]))
	x_bottom,y_bottom = int(float(label[5])),int(float(label[6]))

	x_top_norm, y_top_norm  = x_top/img_w , y_top/img_h
	x_bottom_norm, y_bottom_norm  = x_bottom/img_w , y_bottom/img_h

	x_center_norm = (x_top_norm + x_bottom_norm) / 2
	y_center_norm = (y_top_norm + y_bottom_norm) / 2

	h = abs(x_top_norm-x_bottom_norm)  
	w = abs(y_top_norm-y_bottom_norm)
	with open(os.path.join(save_folder,filename[:-3]+'txt'), "w") as f:
		f.write(f'0 {x_center_norm} {y_center_norm} {w} {h}')

