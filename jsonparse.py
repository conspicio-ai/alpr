import os
import json
import cv2
import numpy as np

# import string
# d = dict.fromkeys(string.ascii_lowercase, 0)
# x ={}
# for i in range(10):
# 	x[str(i)] = i

# for i,n in enumerate(d):
# 	# x[i] =
# 	x[n] =i+10

d = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35}

def bb_test(path):
	image = cv2.imread(path)
	img_h, img_w, _ = image.shape
	x_c, y_c, w, h = 0.11909095636798586, 0.5708955223880597, 0.07384679390503926, 0.5223880597014926
	image = cv2.rectangle(image, (int(img_w*(x_c - w/2)), int(img_h*(y_c - h/2))), (int(img_w*(x_c + w/2)), int(img_h*(y_c + h/2))), [0,0,0], 2) 
	cv2.imshow(' ',image)
	cv2.waitKey(0)


def parser(path):
	"""Returns class, [x_center y_center width height] i.e. Darknet format
	"""
	with open(path) as f:
	  data = json.load(f)

	label_list = []
	coordinates_list = []

	for i in data.get("shapes"):
		image = cv2.imread(path[:-4]+'png')
		img_h, img_w, _ = image.shape
		x, y = np.array(i["points"])
		w, h = y - x
		h /=img_h
		w /=img_w
		x_c, y_c = (x+y)/2
		x_c /=img_w
		y_c /=img_h	
		coordinates_list.append([x_c, y_c, w, h])
		label_list.append(d[(i["label"])])

	return label_list, coordinates_list

def savetxt(dest,path, name):
	"""store to txt file in Darknet format
	"""
	# path_source = os.path.join(dest,path[:-4]+'txt')
	f1 = open(os.path.join(dest,name[:-4]+'txt'),"w") 
	label_list, coordinates_list =  parser(os.path.join(path, name))
	for i,label in enumerate(label_list):
		f1.write("{} {} {} {} {}\n".format(label,coordinates_list[i][0],coordinates_list[i][1],coordinates_list[i][2],coordinates_list[i][3])) 

	f1.close()

# test_path = '/home/rex/projects/number_plate_detection_semantic_segmentation/yolov3/custom/Fiat-Palio-Stile-528458c.jpg_0000_0279_0261_0287_0076.json'
PATH = '/home/rohit/projects/yolo_sih/images'
######### Path is location of image folder which has labels also

lis = os.listdir(PATH)
for i in lis:
	if i.find('.json') != -1:
		print(i+' Doing')
		savetxt('/home/rohit/projects/yolo_sih/label_yolo',PATH, i)
		





# Opening and Closing a file "MyFile.txt" 
# for object name file1. 
