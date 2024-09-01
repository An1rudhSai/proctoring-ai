import cv2
import dlib
import time

# Initialize the Dlib face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor_path = '/Users/amruth/Desktop/Proctoring-AI/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Define the points for the eyes and nose
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
NOSE_TIP = 30

def get_face_landmarks(gray, detector, predictor):
    faces = detector(gray)
    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return landmarks

def detect_significant_head_tilt(landmarks):
    if len(landmarks) == 0:
        return False

    # Extract eye points and nose tip
    left_eye = [landmarks[i] for i in LEFT_EYE_POINTS]
    right_eye = [landmarks[i] for i in RIGHT_EYE_POINTS]
    nose_tip = landmarks[NOSE_TIP]

    # Compute average x-coordinates of the eyes and nose
    left_eye_x = sum([point[0] for point in left_eye]) / len(left_eye)
    right_eye_x = sum([point[0] for point in right_eye]) / len(right_eye)
    nose_x = nose_tip[0]

    # Calculate the tilt angles
    left_eye_nose_diff = nose_x - left_eye_x
    right_eye_nose_diff = right_eye_x - nose_x

    # Define thresholds for significant movement
    left_tilt_threshold = 20  # Threshold for significant left tilt
    right_tilt_threshold = 20  # Threshold for significant right tilt

    # Determine significant head tilt
    if left_eye_nose_diff > left_tilt_threshold and right_eye_nose_diff > left_tilt_threshold:
        return True
    elif left_eye_nose_diff < -right_tilt_threshold and right_eye_nose_diff < -right_tilt_threshold:
        return True
    else:
        return False

# Define mouth opening detection parameters
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0]*5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0]*3

def detect_mouth_opening(shape):
    cnt_outer = 0
    cnt_inner = 0
    for i, (p1, p2) in enumerate(outer_points):
        if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
            cnt_outer += 1 
    for i, (p1, p2) in enumerate(inner_points):
        if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
            cnt_inner += 1
    return cnt_outer > 3 and cnt_inner > 2

def main():
    cap = cv2.VideoCapture(0)
    
    head_tilt_start_time = None
    mouth_open_start_time = None
    head_tilt_total_time = 0
    mouth_open_total_time = 0
    total_start_time = time.time()
    timeout = 60  # Default run time of 60 seconds

    # Record initial mouth distances for calibration
    print("Recording initial mouth distances for calibration...")
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = get_face_landmarks(gray, detector, predictor)
        if landmarks:
            for i, (p1, p2) in enumerate(outer_points):
                d_outer[i] += landmarks[p2][1] - landmarks[p1][1]
            for i, (p1, p2) in enumerate(inner_points):
                d_inner[i] += landmarks[p2][1] - landmarks[p1][1]
    d_outer[:] = [x / 100 for x in d_outer]
    d_inner[:] = [x / 100 for x in d_inner]
    print("Calibration done.")

    # Main loop for detecting mouth opening and head tilt
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = get_face_landmarks(gray, detector, predictor)
        
        current_time = time.time()

        if current_time - total_start_time >= timeout:
            break
        
        if landmarks:
            is_significant_tilt = detect_significant_head_tilt(landmarks)
            if is_significant_tilt:
                if head_tilt_start_time is None:
                    head_tilt_start_time = current_time
            else:
                if head_tilt_start_time is not None:
                    head_tilt_total_time += current_time - head_tilt_start_time
                    head_tilt_start_time = None

            if detect_mouth_opening(landmarks):
                if mouth_open_start_time is None:
                    mouth_open_start_time = current_time
            else:
                if mouth_open_start_time is not None:
                    mouth_open_total_time += current_time - mouth_open_start_time
                    mouth_open_start_time = None
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finalize head tilt and mouth open times
    if head_tilt_start_time is not None:
        head_tilt_total_time += time.time() - head_tilt_start_time
    if mouth_open_start_time is not None:
        mouth_open_total_time += time.time() - mouth_open_start_time

    total_time = time.time() - total_start_time
    head_tilt_total_time = total_time - head_tilt_total_time
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Total time run: {total_time:.2f} seconds")
    print(f"Total time head was tilted: {head_tilt_total_time/3:.2f} seconds")
    print(f"Total time mouth was open: {mouth_open_total_time:.2f} seconds")

if __name__ == "__main__":
    main()
