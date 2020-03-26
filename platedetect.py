import cv2
import numpy as np
#from pytesseract import pytesseract as tess
#from re import compile
import re
import imutils
from imutils import contours
from perspectivecorrector import *
import time
import sys
import scipy.fftpack
#Setting the Indian Number Plate Format for filtering using Regular Expressions (regex)

plate_format = '[A-Z]{2}.*[0-9]{2}.*[A-Z]{0,3}.*[0-9]{4}$'

#Function to filter irrelevent number plate detection strings that do not satisfy the expression
def IoU(box1,box2):
	# Step 1: Finding the intersection area
	x1 = max(box1[0],box2[0])
	x2 = min(box1[2],box2[2])
	y1 = max(box1[1],box2[1])
	y2 = min(box1[3],box2[3])
	interArea = max(0,x2 - x1 + 1) * max(0,y2 - y1 + 1)
	box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
	box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
	iou = interArea/float(box1Area + box2Area - interArea)
	if min(box1Area,box2Area) == interArea:
		return -1.0
	else:
		return iou
	

def get_contour_precedence(contour,method = "left-to-right"):
	boundingBoxes = [cv2.boundingRect(c) for c in contour]
	(contour, boundingBoxes) = zip(*sorted(zip(contour,boundingBoxes),key = lambda b: b[1][0], reverse = False))
	return (contour, boundingBoxes) 
	

def imclearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    im2,contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy


def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    im2,contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy


def plate_status(x):
	x = re.findall(plate_format,x,re.MULTILINE)
	x =''.join(x)
	x = re.sub('[^A-Za-z0-9]+', '', x)
	y = re.sub('\s*\'*[a-z]*','',x)
	return y
	
# The main function, taking input as the image and output as the number plate.
# original - original feed, rf - resize_factor of the image, img - image on which semantic segmentation is applied
def img2str(original,rf,img):
	alphanumerics = []
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
		
		#return plate
		
		# Applying FFT for number plate denoising (clipping using low and high pass filter with the required Gaussian Parameters
		img = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
		# Number of rows and columns
		rows = img.shape[0]
		cols = img.shape[1]

		rows = img.shape[0]
		cols = img.shape[1]

		# Convert image to 0 to 1, then do log(1 + I)
		imgLog = np.log1p(np.array(img, dtype="float") / 255)

		# Create Gaussian mask of sigma = 10
		M = 2*rows + 1
		N = 2*cols + 1
		sigma = 5
		(X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
		centerX = np.ceil(N/2)
		centerY = np.ceil(M/2)
		gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

		# Low pass and high pass filters
		Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
		Hhigh = 1 - Hlow

		# Move origin of filters so that it's at the top left corner to
		# match with the input image
		HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
		HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

		# Filter the image and crop
		If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
		Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
		Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

		# Set scaling factors and add
		gamma1 = 0.3
		gamma2 = 1.5
		Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]

		# Anti-log then rescale to [0,1]
		Ihmf = np.expm1(Iout)
		Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
		Ihmf2 = np.array(255*Ihmf, dtype="uint8")

		# Threshold the image - Anything below intensity 65 gets set to white
		Ithresh = Ihmf2 < 65
		Ithresh = 255*Ithresh.astype("uint8")

		# Clear off the border.  Choose a border radius of 5 pixels
		Iclear = imclearborder(Ithresh, 5)
		#Iclear = Ithresh
		# Eliminate regions that have areas below 50 pixels
		Iopen = bwareaopen(Iclear, 50)
		kernel = np.ones((3,3), np.uint8)
		Iopen = cv2.morphologyEx(Iopen, cv2.MORPH_CLOSE, kernel)
		#Iopen = cv2.morphologyEx(Iopen, cv2.MORPH_OPEN, kernel)
		im2,cnts,hierarchy = cv2.findContours(Iopen,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		#cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
		if len(cnts) != 0:
			(cnts, boundingBoxes) = get_contour_precedence(cnts)
		#contours = sorted(contours,key = cv2.contourArea)
		Iopen = cv2.cvtColor(Iopen,cv2.COLOR_GRAY2BGR)
		i = 0
		#print("Iteration = ",counter," length = ",len(cnts))
		#print('----------')
		while i < len(cnts):
			#Iopen = cv2.putText(Iopen, str(i), cv2.boundingRect(cnts[i])[:2],cv2.FONT_HERSHEY_SIMPLEX,1,[125])
			cnt = cnts[i]
			if i == 0:
				x0,y0,w0,h0 = cv2.boundingRect(cnts[0])
				cv2.rectangle(Iopen,(x0,y0),(x0+w0,y0+h0),(0,255,0),1)
				alphanumerics = alphanumerics + [[(x0,y0),(x0+w0,y0+h0)]]	
				i = i + 1
				continue		
			x,y,w,h = cv2.boundingRect(cnt)
			#print(w*h)
			iou = IoU((x0,y0,x0+w0,y0+h0),(x,y,x+w,y+h))
			#print(iou)
			#iou = 0.0
			if iou == -1:
				if (w0*h0) > (w*h):
					i = i + 1
					continue
				else:
					cv2.rectangle(Iopen,(x,y),(x+w,y+h),(0,255,0),1)
					alphanumerics = alphanumerics + [[(x,y),(x+w,y+h)]]
			elif iou == 0:
				cv2.rectangle(Iopen,(x,y),(x+w,y+h),(0,255,0),1)
				alphanumerics = alphanumerics + [[(x,y),(x+w,y+h)]]
			else:
				if iou < 0.1:
					cv2.rectangle(Iopen,(x,y),(x+w,y+h),(0,255,0),1)
					alphanumerics = alphanumerics + [[(x,y),(x+w,y+h)]]
				else:
					i = i + 1
					continue
			x0 = x
			y0 = y
			w0 = w
			h0 = h		
			i = i + 1
			
		return original, Iopen, alphanumerics
		#cv2.imshow('Overall Result', Iopen)
		#cv2.imshow('Original Image',original)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		#counter = counter + 1
		#Image Enhancement on segmented number plate

		#gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
		#ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


		#tessdetect = tess.image_to_string(thresh, config='eng')
		#text = plate_status(tessdetect)
		#return original,thresh,text

	else:
		return original, np.zeros((50,100)), 'Nothing found.'
		

