import cv2
import numpy as np

# Load the pre-trained DNN model for face detection
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Initialize the webcam
webcam = cv2.VideoCapture(0)

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    return gray_frame

def detect_smile(face_roi_gray):
    smiles = smile_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.7, minNeighbors=30, minSize=(25, 25))
    return len(smiles) > 0

while True:
    # Read the current frame from the webcam
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break

    # Get the frame dimensions
    (h, w) = frame.shape[:2]

    # Preprocess the frame for the DNN model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Detect faces using the DNN model
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x1, y1), (100, 200, 50), 4)

            # Define the region of interest (ROI) for smile detection within the face
            face_roi = frame[y:y1, x:x1]
            face_roi_gray = preprocess_frame(face_roi)

            # Detect smiles in the face ROI
            if detect_smile(face_roi_gray):
                cv2.putText(frame, 'Smiling', (x, y1 + 40), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # Display the frame with face and smile detection
    cv2.imshow('Real-time Face and Smile Detection', frame)

    # Break out of the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
webcam.release()
cv2.destroyAllWindows()
