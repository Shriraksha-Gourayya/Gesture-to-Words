import os
import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from keras.models import load_model


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gesture_model.h5"
)

CLASSES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "classes.npy"
)


# ============================================================
# SETTINGS
# ============================================================

SEQ_LEN = 30
FEATURES = 1629

# Camera 0 = the phone camera feed confirmed working in the test
CAMERA_INDEX = 0


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found:")
    print(MODEL_PATH)
    raise SystemExit

if not os.path.exists(CLASSES_PATH):
    print("ERROR: Classes file not found:")
    print(CLASSES_PATH)
    raise SystemExit


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# LOAD CLASSES
# ============================================================

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

print("Loaded classes:")
print(classes)

print("Number of classes:", len(classes))


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract(results):

    # --------------------------------------------------------
    # Face landmarks
    # --------------------------------------------------------

    if results.face_landmarks:

        face = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.face_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        face = np.zeros(
            468 * 3,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Pose landmarks
    # --------------------------------------------------------

    if results.pose_landmarks:

        pose = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.pose_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        pose = np.zeros(
            33 * 3,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Left hand landmarks
    # --------------------------------------------------------

    if results.left_hand_landmarks:

        left = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.left_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        left = np.zeros(
            21 * 3,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Right hand landmarks
    # --------------------------------------------------------

    if results.right_hand_landmarks:

        right = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.right_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        right = np.zeros(
            21 * 3,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Combine all features
    # --------------------------------------------------------

    features = np.concatenate(
        [
            face,
            pose,
            left,
            right
        ]
    )

    return features.astype(np.float32)


# ============================================================
# OPEN PHONE CAMERA FEED
# ============================================================

print("\nOpening phone camera feed...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print("ERROR: Could not open the phone camera feed.")
    print("Camera index:", CAMERA_INDEX)
    raise SystemExit


# ============================================================
# CAMERA SETTINGS
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ============================================================
# CAMERA WARM-UP
# ============================================================

print("Waiting for phone camera feed to initialize...")

time.sleep(2)


frame = None
ret = False


for attempt in range(30):

    ret, frame = cap.read()

    if ret and frame is not None:

        print(
            "Phone camera feed ready after",
            attempt + 1,
            "attempt(s)."
        )

        break

    print(
        f"Waiting for camera frame... "
        f"attempt {attempt + 1}/30"
    )

    time.sleep(0.1)


if not ret or frame is None:

    print()
    print("ERROR: Camera opened but no frame was received.")
    print()
    print("Make sure:")
    print("1. Keep Windows phone-camera sharing enabled.")
    print("2. Windows Camera is not being used by another app.")
    print("3. Camera permission is enabled.")
    print()

    cap.release()

    raise SystemExit


# ============================================================
# FRAME BUFFER
# ============================================================

buffer = deque(
    maxlen=SEQ_LEN
)


# ============================================================
# START MEDIAPIPE HOLISTIC
# ============================================================

print("Starting MediaPipe Holistic...")


with mp_holistic.Holistic(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    print("MediaPipe Holistic started.")

    print()
    print("==============================================")
    print("          ISL WORD RECOGNITION")
    print("==============================================")
    print("Camera: Phone camera feed (index 0)")
    print("Show an ISL gesture to the camera.")
    print("Press Q to quit.")
    print("==============================================")
    print()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()


        # ----------------------------------------------------
        # Camera frame failure
        # ----------------------------------------------------

        if not ret or frame is None:

            continue


        # ----------------------------------------------------
        # Mirror the camera feed
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # Convert BGR to RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MediaPipe processing
        # ----------------------------------------------------

        results = holistic.process(
            rgb
        )


        # ====================================================
        # DRAW FACE LANDMARKS
        # ====================================================

        if results.face_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS
            )


        # ====================================================
        # DRAW POSE LANDMARKS
        # ====================================================

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )


        # ====================================================
        # DRAW LEFT HAND LANDMARKS
        # ====================================================

        if results.left_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )


        # ====================================================
        # DRAW RIGHT HAND LANDMARKS
        # ====================================================

        if results.right_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )


        # ====================================================
        # EXTRACT FEATURES
        # ====================================================

        features = extract(
            results
        )


        # ----------------------------------------------------
        # Verify feature size
        # ----------------------------------------------------

        if features.shape[0] != FEATURES:

            print(
                "ERROR: Feature size mismatch."
            )

            print(
                "Received:",
                features.shape[0]
            )

            print(
                "Expected:",
                FEATURES
            )

            continue


        # ====================================================
        # ADD FRAME TO BUFFER
        # ====================================================

        buffer.append(
            features
        )


        # ====================================================
        # DEFAULT MESSAGE
        # ====================================================

        text = "Show a gesture..."


        # ====================================================
        # PREDICTION AFTER 30 FRAMES
        # ====================================================

        if len(buffer) == SEQ_LEN:

            sequence = np.array(
                buffer,
                dtype=np.float32
            )


            sequence = sequence.reshape(
                1,
                SEQ_LEN,
                FEATURES
            )


            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            predictions = model.predict(
                sequence,
                verbose=0
            )[0]


            # ------------------------------------------------
            # Find highest probability
            # ------------------------------------------------

            index = int(
                np.argmax(predictions)
            )


            confidence = (
                float(predictions[index]) * 100
            )


            # ------------------------------------------------
            # Get predicted class
            # ------------------------------------------------

            label = str(
                classes[index]
            ).strip()


            # ------------------------------------------------
            # Prediction text
            # ------------------------------------------------

            text = (
                f"{label} "
                f"({confidence:.2f}%)"
            )


        # ====================================================
        # PREDICTION DISPLAY BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (700, 75),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        cv2.putText(
            frame,
            text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # DISPLAY CAMERA FEED
        # ====================================================

        cv2.imshow(
            "ISL Word Recognition - Press Q to Quit",
            frame
        )


        # ====================================================
        # QUIT WITH Q
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("ISL Word Recognition stopped.")