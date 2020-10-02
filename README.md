# Number Plate Detection
![http://ivlabs.in/](https://raw.githubusercontent.com/IvLabs/resources/master/ivlabs-black.png)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/IvLabs/resources/blob/master/LICENSE.md)


<p align="center">
  <img width="200" height="200" src="images/Falcon_Vision_logo.jpg">
</p>

<details>
<summary>Table of content (Click Here to Expand)</summary>

- [Number Plate Detection](#number-plate-detection)
  - [Important Link](#important-link)
  - [Initial Setup](#initial-setup)
  - [How to Run the Pipeline](#how-to-run-the-pipeline)
  - [Dataset Info](#dataset-info)
  - [Problem Statement](#problem-statement)
  - [Pipeline](#pipeline)
    - [Vehicle Type Detection](#vehicle-type-detection)
      - [Detected Classes:](#detected-classes)
    - [Plate Detection](#plate-detection)
    - [Alphanumeric Digits Detection](#alphanumeric-digits-detection)
    - [Convert to String](#convert-to-string)
    - [Data to Cloud](#data-to-cloud)
    - [Web App Integration](#web-app-integration)
    - [Android App Integration](#android-app-integration)
  - [Contributors](#contributors)
 
</details>


## Important Link
- **PPT Link:** 
  - Main PPT: [Google Slides](https://docs.google.com/presentation/d/1q39IvLVMG8u-abPxm9u3NtT5gNQdbnQkPM5l6BtNbTA/edit?usp=sharing)
  - Day 2 PPT: [Google Slides](https://docs.google.com/presentation/d/1qU8sJP_Nky-A7zDzgn7S3SJg2Id4tf-d4mEfXrIsbyw/edit?usp=sharing)
- **YouTube Video:** [Working Demo](https://youtu.be/Y-EuzsubixI)
- **Business Plan:** [Google Docs](https://docs.google.com/document/d/1aqqrfF1NUa2LrXZRQiwlgdjGuqVql-PC6bh0hWQVHKw/edit?usp=sharing)
- **WebApp:** [GitHub Repo](https://github.com/rishesh007/falcon-vision-web-app)
- **Android App:** [GitHub Repo](https://github.com/ashukin/CB31_IvLabs_WIMDR_AndroidApp)
- **Sample Test results:** [Videos Folder](https://drive.google.com/drive/folders/10Dy622z1tN7ow2a7XdfU7zSwE7M2PBao?usp=sharing)


## Initial Setup

1. Clone this repository `git clone https://github.com/take2rohit/CB31_IvLabs_WIMDR.git`
2. Change your directory `cd CB31_IvLabs_WIMDR`
3. Make your virtual environment either using virtualenv or conda `virtualenv sih -p python3` (optional)
4. Activate the virtual environment `source sih/bin/activate` (optional)
5. Install all dependencies `pip3 install -r requirements.txt` 
6. Make sure all the NVIDIA Drivers are installed for best performance
7. Download three weights and place them in `CB31_IvLabs_WIMDR/weights/` folder.
   - OCR detection: [Google Drive](https://drive.google.com/file/d/1IvqQo5HGZiWecVytStlm25qVRcUXmK5m/view?usp=sharing)
   - Vehicle Detection: [Google Drive](https://drive.google.com/file/d/1ffX18zypBmMs9K9YSaAZHieyNIxN-NMB/view?usp=sharing)
   -  Number Plate: [Google Drive](https://drive.google.com/file/d/1AZmdRm8YurVV3v_fNV79bzyXCLcatMJe/view?usp=sharing)

## How to Run the Pipeline
   
1. Open the file `yolov4_test.py` locate call of `main()` function (At the end of the file).
2. Change your video file name in `video_file` variable.
3. Open terminal and run `python3 yolov4_test.py`
4. If you are not getting required output try changing hyper-parameters of code 
   
***Note***: 
- In case of `RuntimeError: CUDA out of memory.` please change variable `use_cuda` as `use_cuda=False`
- If the code not working try changing hyperparameters like `window_size`, `frame_skipping`, `confidence`, etc.

## Dataset Info
> Total size of all database = 10 GB+
- Manually labelled Vehicle and Plate Dataset: [Link](https://drive.google.com/file/d/1KIUP5lcwD2J_Qru84fpO4zIHQVvXKwkq/view?usp=sharing) 
- Manually labelled OCR dataset: [Link](https://drive.google.com/drive/folders/13TIxYQ4MRUCUbSY-PMXHMHaglLVCLy7l?usp=sharing)
- Manually labelled OCR dataset (MixOrg): [Link](https://drive.google.com/drive/folders/1dqzcRnBm0PEmQFhIvBn693bmY3s9zSEj?usp=sharing)
- Manually labelled License Plate Dataset: [Link](https://drive.google.com/drive/folders/1m8BC-KCPNyNFf-g39ZV9cAZ-PjdI9NPw?usp=sharing)
- Pre-labelled OpenImage Dataset: [Link](https://storage.googleapis.com/openimages/web/index.html)



## Problem Statement

1. Localities face persistent threat of security due to illegal parking, theft and unregulated entry/exit.
2. We aim to automate the registration of entry/exits of every commute to ensure round the clock surveillance.
3. Database with registered entry can be used to prevent theft.
4. Can be extended for parking management in malls, shopping complexes, theatres, etc.

**Problem Statement ID:** CB31
**Problem Statement Organizer:** [MixORG](https://mixorg.com/)

## Pipeline

1. [Vehicle Type Detection](#Vehicle-Type-Detection): Classifying the vehicle, whether it is a car, bus, truck, etc.
2. [Plate Detection](#Plate-Detection): Segmenting the number plate.
3. [Alphanumeric Digits Detection](#Alphanumeric-Digits-Detection): Segmenting the number plate digits from the number plate.
4. [Convert to String](#Convert-to-String): Each digit is concatenated to form a string.
5. [Data to Cloud](#Data-to-Cloud): When the final string is obtained, it is pushed in the cloudbase.
6. [Web App Integration](#Web-App-Integration): Vehicle information shall be visible on the web application to the admin. More information can be found [here](https://github.com/rishesh007/falcon-vision-web-app)
7. [Android App Integration](#Android-App-Integration): A user - friendly android app, rich in features is also developed. More information can be found [here](https://github.com/ashukin/CB31_IvLabs_WIMDR_AndroidApp)
8. Deployment: Integration of steps 1 to 6 for realtime deployment in different scenarios.


<p align="center">
  <img src="images/system_design.svg">
</p>

### Vehicle Type Detection
First the image is acquired through a camera placed at an optimal position from where the vehicle and number plate are recognizable. Starting with vehicle type detection, it detect the type of the vehicle such as whether it is a car, truck, bicycle, etc. (YOLOv4 has been used, trained on COCO dataset for vehicles).

#### Detected Classes:
* Car
* Motorbike
* Truck
* Number Plate

### Plate Detection

<p align="center">
  <img src="/images/platedetect.gif">
</p>

The step focusses on the detection of number plate, which was trained with some of our own dataset (gathered and annotated by us) as well as some from of Open Images Dataset. This was trained with YOLOv4 architecture. The results we got were not only robust but also accurate.

### Alphanumeric Digits Detection

<p align="center">
  <img src="/images/plate.gif">
</p>

The number plate detected is now segmented and passed through another YOLOv4 architecture (multiclass classification), trained again on some of our own dataset as well as with some dataset from Kaggle. In order to incorporate the number plates with more than one line (such as on trucks and buses), we perform affine correction to make the plate symmetric about x and y axis (make it rectangular). Now, the plate is read from left to right and top to bottom.

### Convert to String

<p align="center">
  <img src="/images/string.gif">
</p>

The digits are read from the previous step and coverted to string.

### Data to Cloud

The obtained string is then pushed to the firebase, which links the final web integration to record the dynamic changes. Each number plate is mapped to the number of visits made through a particular gate or society and also whether the vehicle is authorized to pass the gate or not. 

**Realtime Implementation:** [Google Drive](https://drive.google.com/file/d/1Q7oO0bE7CAbIwE7LhOG1o1FxsaqpTDkZ/view?usp=sharing)

### Web App Integration

The details about any activity of user's car will be updated on the web app. The administrator can monitor the traffic flow and prevent blacklisted vehicles from passing. For every entry/exit of vehicle, its snapshot will be processed and stored, for future reference.

**Realtime Implementation:** [Google Drive](https://drive.google.com/file/d/10C3a_05qDVWPd9jqBfU9TY7l8JhiE9LL/view?usp=sharing)

### Android App Integration

A user - friendly android app has been created, so the user can register his/her vehicle and link with their car. The car can be registered easily by scanning a QR code maintained by the society. Other features include an interactive dashboard, UPI payment for parking charges, notice board for society, vigilance mode to send notification about any activity related to the user's vehicle and provision to add more than one vehicle per user. 

**Realtime Implementation:** [Google Drive](https://drive.google.com/file/d/1v1akjvle2rp9KxP0MkASVxgWkcBIhED2/view)



## Contributors
- *Rohit Lal* - [website](http://take2rohit.github.io/) 
- *Himanshu Patil* - [LinkedIn](https://www.linkedin.com/in/hipatil/)
- *Khush Agrawal* - [website](https://khush3.github.io/)
- *Rishesh Agarwal* - [LinkedIn](https://www.linkedin.com/in/rishesh-agarwal/)
- *Arihant Gaur* - [website](https://flagarihant2000.github.io/arihantgaur/)
- *Akshata Kinage* - [LinkedIn](https://www.linkedin.com/in/akshatakinage/)
