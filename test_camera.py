import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera opened:", cap.isOpened())

if cap.isOpened():
    print("Waiting for camera...")
    time.sleep(2)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to receive frame")
            break

        cv2.imshow("Live Camera Check - Press Q to Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("Camera could not be opened.")