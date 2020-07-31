# Number Plate Detection
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

<p align="center">
  <img width="500" height="500" src="images/Falcon_Vision_logo.jpg">
</p>

## Problem Statement

1. Localities face persistent threat of security due to illegal parking, theft and unregulated entry/exit.
2. We aim to automate the registration of entry/exits of every commute to ensure round the clock surveillance.
3. Database with registered entry can be used to prevent theft.
4. Can be extended for parking management in malls, shopping complexes, theatres, etc.

**Problem Statement ID:** CB31
**Problem Statement Organizer:** [MixORG](https://mixorg.com/)

## Pipeline

1. [Vehicle Type Detection](##Vehicle-Type-Detection): Classifying the vehicle, whether it is a car, bus, truck, etc.
2. [Plate Detection](##Plate-Detection): Segmenting the number plate.
3. [Alphanumeric Digits Detection](##Alphanumeric-Digits-Detection): Segmenting the number plate digits from the number plate.
4. [Convert to String](##Convert-to-String): Each digit is concatenated to form a string.
5. [Data to Cloud](##Data-to-Cloud): When the final string is obtained, it is pushed in the cloudbase.
6. [Web App Integration](##Web-App-Integration): Vehicle information shall be visible on the web application. More information can be found [here](https://github.com/rishesh007/falcon-vision-web-app)
7. Deployment: Integration of steps 1 to 6 for realtime deployment in different scenarios.



<p align="center">
  <img src="images/system_design.svg">
</p>

### Vehicle Type Detection
`Add vehicle type detection .gif here`

`Add Description`

### Plate Detection

<p align="center">
  <img src="/images/platedetect.gif">
</p>

`Add Description`

### Alphanumeric Digits Detection

<p align="center">
  <img src="/images/plate.gif">
</p>
`Add Description`

### Convert to String

<p align="center">
  <img src="/images/string.gif">
</p>

`Add Description`

### Data to Cloud

Owing to the large size of the graphic, the `.gif` file can be seen [here](https://drive.google.com/file/d/1d5iCLz8caoTTKYYxlhKmf1rZWejHspYS/view?usp=sharing)
`Add Description`

### Web App Integration

Owing to the large size of the graphic, the `.gif` file can be seen [here](https://drive.google.com/file/d/1BMA6nsOvdXIhl4wpWVG8JOcQKpyvZRZE/view?usp=sharing)

## Installation Guide (for Ubuntu)

1. Clone this repository `git clone https://github.com/take2rohit/sih_number_plate.git`
2. Change your directory `cd sih_number_plate`
3. Make your virtual environment `virtualenv sih -p python3` (optional)
4. Activate the virtual environment `source sih/bin/activate` (optional)
5. Install all dependencies `pip3 install -r requirements.txt` 
6. Make sure all the NVIDIA Drivers are installed for best performance

## How to Run the Pipeline

1. Download three weights from [here](), [here]() and [here](). Make sure that they are present in the current directory.
2. Execute the command `python3 yolov4_test.py`. Enter the directory location for the video to be tested. Two pop up videos appear, showing the annotations on vehicle and number plate, along with text displayed, indicating various parameters such as accuracy of number plate detection, vehicle class, type of vehicle (commercial, private or other type) and the FPS for the current feed.
3. Video results are stored in `plate_result.avi` for annotated video feed and `digit_result.avi` for segmented number plate with digit annotations.

## How to Run the Web Application

`Write the procedure here`

### Contributors
- Rohit Lal 
- Himanshu Patil
- Khush Agrawal
- Rishesh Agarwal
- Arihant Gaur
- Akshata Kinage
