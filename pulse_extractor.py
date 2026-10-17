import cv2
import mediapipe as mp
import numpy as np

def extract_raw_signal(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        refine_landmarks=False, 
        min_detection_confidence=0.5
    )
    
    forehead_indices = [10, 107, 66, 69, 109, 338, 296, 299, 336]
    raw_signal = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            
            roi_pixels = []
            for idx in forehead_indices:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                if 0 <= x < w and 0 <= y < h:
                    roi_pixels.append(frame[y, x, 1]) # Green channel
            
            if roi_pixels:
                raw_signal.append(np.mean(roi_pixels))
        else:
            raw_signal.append(raw_signal[-1] if raw_signal else 0)
            
    cap.release()
    return np.array(raw_signal), fps