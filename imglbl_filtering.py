import os
CONVERT = False
# classes = ['Ambulance','Bicycle','Bus','Car','Limousine','Motorcycle','Taxi','Truck','Van']
classes = ['Bus']

for c in classes:
	lbl = []
	img =[]
	print(c)
	lbl_dir = '/home/himanshu/Downloads/train/Label'
	img_dir = '/home/himanshu/Downloads/train/Vehicle_registration_plate'
	if CONVERT == True:
		for lbl_file in os.listdir(lbl_dir):
			if not os.path.exists(os.path.join(img_dir,lbl_file[:-3]+'jpg')):
				os.remove(os.path.join(lbl_dir,lbl_file))

	else:
		lbl = list(os.listdir(lbl_dir))
		label = []
		for l in lbl:
			label.append(l[:-4])

		image = []
		img = list(os.listdir(img_dir))
		for m in img:
			image.append(m[:-4])

		image.sort()
		label.sort()

		for i, j in zip(image,label):
			if i != j:
				print(i,j)

# print(label)
# print(image)
# if label is image:
# 	print("SAME!!!!!!")

