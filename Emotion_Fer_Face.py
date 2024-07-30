from fer import FER
import cv2

# Initialize the FER detector
detector = FER()

# Initialize webcam
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break

    # Analyze emotions
    emotion_analysis = detector.detect_emotions(frame)
    for face in emotion_analysis:
        (x, y, w, h) = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        emotion = max(face['emotions'], key=face['emotions'].get)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
