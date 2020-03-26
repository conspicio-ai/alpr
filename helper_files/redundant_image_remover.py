import os 

root = 'dataset'
img = 'images'
labels = 'labels'
images_files = os.listdir(os.path.join(root,img))

for i in images_files:
	if i[:-3]+'txt' not in os.listdir(os.path.join(root,labels)):
		os.remove(os.path.join(root,img,i))
		print('removed: '+ i)