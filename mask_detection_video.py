# Importing Libraries
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


# Define input and video path
input_video_path = 'videos/Face_Mask.mp4'
output_video_path = 'videos/output.mp4'

# Loading the Model
model = load_model('models/mask_detector.h5')

# Loading the face detection model
net = cv2.dnn.readNetFromCaffe("models/weights-prototxt.txt", "models/res_ssd_300Dim.caffeModel")

# Function to get the bounding boxes from the detections
def get_detected_regions(detections):
    filtered_detections = []
    for i in range(0, detections.shape[2]):
        # print("detections....")
        confidence = detections[0, 0, i, 2]
        # greater than the minimum confidence
        if confidence > 0.2:
            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            (x1, y1, x2, y2) = box.astype("int")
            filtered_detections.append([x1, y1, x2, y2])
            # print(x1, y1, x2, y2)
    return np.array(filtered_detections)

# Get the region of interest (detected faces)
def get_detected_faces(detections):
    faces_images = []
    for detection in detections:
        x1, y1, x2, y2 =  detection
        cropped_faces = frame[y1:y2, x1:x2]
        cropped_faces = cv2.cvtColor(cropped_faces, cv2.COLOR_BGR2RGB)
        cropped_faces = cv2.resize(cropped_faces, (224, 224))
        cropped_faces = img_to_array(cropped_faces)
        faces_images.append(cropped_faces) 

    return faces_images

# Read the video
video_capture = cv2.VideoCapture(input_video_path)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fps = video_capture.get(cv2.CAP_PROP_FPS) # To get the frame rate of the video

# Define the codec and create VideoWriter object.in 'outpy.avi' file.
out = cv2.VideoWriter(output_video_path,
        cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

while True:
    # Read the video frames
    _, frame = video_capture.read()
    # Get the height and width of the frame
    # (height, width) = frame.shape[:2]
    # Applying the face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob into dnn 
    net.setInput(blob)
    detections = net.forward()
    # print(detections)

    try:
        detections = get_detected_regions(detections)
        # print("Detections:",detections)
        faces = get_detected_faces(detections)
        faces = np.array(faces)
        face = preprocess_input(faces)
        preds = model.predict(face)

        correct_mask_count = []
        incorrect_mask_count = []
        no_mask_count = []

        i = 0
        for pred in preds:
            (WithoutMask, CorrectMask, InCorrectMask) = pred
            if max(pred) == CorrectMask:
                label = " Correct Mask"
                color = (0, 255, 0)
                correct_mask_count.append(1)
            elif max(pred) == InCorrectMask:
                label = " Incorrect Mask"
                color = (250, 00, 0)
                incorrect_mask_count.append(2)
            else:
                label = " No Mask"
                color = (0, 0, 255)
                no_mask_count.append(0)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(WithoutMask, CorrectMask, InCorrectMask) * 100)
            (x1, y1, x2, y2) = detections[i]

            # Displaying the labels
            # cv2.rectangle(frame, (x1, y1 + 20), (x2+5, y2+15), color, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, label, (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            i += 1

        face_count = len(no_mask_count) + len(incorrect_mask_count) + len(correct_mask_count)

        text = "FaceCount: {}   NoMaskCount: {}   CorrectMaskCount: {}  InCorrectMaskCount: {}".format(
            face_count, len(no_mask_count), len(correct_mask_count), len(incorrect_mask_count))
        cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    except:
        # print("error...")
        pass

    
    # Write the frame into the file 'output.avi'
    out.write(frame)

    cv2.imshow('FACE MASK DETECTOR', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release video capture
video_capture.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows()
