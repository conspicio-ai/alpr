import cv2
import numpy as np
from pytesseract import pytesseract as tess
#from re import compile
import re
import imutils
from perspectivecorrector import *
import time
import sys

#Setting the Indian Number Plate Format for filtering using Regular Expressions (regex)

plate_format = '[A-Z]{2}.*[0-9]{2}.*[A-Z]{0,3}.*[0-9]{4}$'

#Function to filter irrelevent number plate detection strings that do not satisfy the expression
def plate_status(x):
	x = re.findall(plate_format,x,re.MULTILINE)
	x =''.join(x)
	x = re.sub('[^A-Za-z0-9]+', '', x)
	y = re.sub('\s*\'*[a-z]*','',x)
	return y
	
# The main function, taking input as the image and output as the number plate.
# original - original feed, rf - resize_factor of the image, img - image on which semantic segmentation is applied
def img2str(original,rf,img):

	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))

	kernel = np.ones((5,5),np.uint8)

	sh = img.shape

	#original = cv2.resize(original,(sh[1],sh[0]))
	#Image enhancement using morphological transformation
	ret,thresh = cv2.threshold(img,60,255,0)
	thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)

	#Contour detection for detecting the number plate area
	contours,hierarchy = cv2.findContours(thresh,1,2)

	if len(contours) >0:
		c = max(contours, key=cv2.contourArea)
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])
		rect = cv2.minAreaRect(c)
		pts11 = cv2.boxPoints(rect)
		box = np.int0(pts11)*rf
		p1,p2,p3,p4 = box

		# Contour detection for semantic segmented plate area
		cv2.drawContours(original, [box], 0, (36,255,12), 4)


		pts = np.array([p1, p2, p3, p4], dtype = "float32")

		# Applying perspective transform to change the viewpoint of the number plate for more accurate detection
		plate = four_point_transform(original,pts)

		#Image Enhancement on segmented number plate

		gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


		#c = time.time()
		# Use of pytesseract for detecting the string from the segmented binarized number plate
		tessdetect = tess.image_to_string(thresh, config='eng')
		# tessdetect = 'ads'
		#Apply filtering on the detected string, so as to output the relevent number plate string.
		text = plate_status(tessdetect)
		return original,thresh,text

	else:
		return original, np.zeros((50,100)), 'Nothing found.'
