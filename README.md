# mask_detection

## Introduction

Design and develop an application having the following functionality in live video.
1. If no person is in the video, it should alert “No Person”.
2. If the person walks in, detect the person and assign a unique ID.
3. Detect the person wearing a mask or not with alerts like “Mask Detected” and “No Mask
Detected”.
4. If a new person walks in, repeat steps 2 and 3 for him/her.
5. If the two persons are standing close enough, it should alert ”Maintain Social Distancing”.

## Requirements/Dependencies

Please run requirement.txt. It would install all the requirements.

## Input 

Create videos folder and put the input video (Face_Mask.mp4) here, which you can download from "https://drive.google.com/drive/folders/1yVKIPn2pKTbssX5X68qdfzD2_swC7HTd?usp=share_link"

## Model

download the model (mask_detector.h5) "https://drive.google.com/drive/folders/1yVKIPn2pKTbssX5X68qdfzD2_swC7HTd?usp=share_link" and put it into models folder.

## How to run

run python mask_detection_video.py

it would generate a video which detects the mask or no mask with person. 

## Results

Output video (output.avi) which shows the result "https://drive.google.com/drive/folders/1yVKIPn2pKTbssX5X68qdfzD2_swC7HTd?usp=share_link"


