import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLO-family pose estimation model
@st.cache_resource
def load_model():
    # Load YOLOv8 pose estimation model
    model = YOLO("model.pt")  # Ensure you have the correct path to your model
    return model

# Load face detection model
@st.cache_resource
def load_face_model():
    # Option 1: Use YOLOv8 face detection model (download automatically)
    try:
        face_model = YOLO("yolov8n-face.pt")
        return face_model, "yolo"
    except:
        # Option 2: Fallback to OpenCV Haar Cascade (built-in)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade, "opencv"

def detect_faces(frame, face_model, model_type="yolo"):
    """
    Detect faces in the frame using either YOLO or OpenCV
    
    Args:
        frame: Input image frame
        face_model: Face detection model
        model_type: "yolo" or "opencv"
    
    Returns:
        List of face bounding boxes [(x1, y1, x2, y2), ...]
    """
    faces = []
    
    if model_type == "yolo":
        # Use YOLO face detection
        results = face_model(frame)
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    if conf > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, box)
                        faces.append((x1, y1, x2, y2))
    
    elif model_type == "opencv":
        # Use OpenCV Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_model.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in detected_faces:
            faces.append((x, y, x + w, y + h))
    
    return faces

def draw_face_boxes(frame, faces):
    """
    Draw bounding boxes around detected faces
    
    Args:
        frame: Input image frame
        faces: List of face bounding boxes
    
    Returns:
        Frame with face boxes drawn
    """
    for i, (x1, y1, x2, y2) in enumerate(faces):
        # Draw face bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add label - showing "Happy" for fake data
        cv2.putText(frame, "Happy", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return frame

def extract_face_region(frame, faces, face_index=0):
    """
    Extract face region from frame
    
    Args:
        frame: Input image frame
        faces: List of face bounding boxes
        face_index: Which face to extract (default: first face)
    
    Returns:
        Cropped face image or None if no face found
    """
    if len(faces) > face_index:
        x1, y1, x2, y2 = faces[face_index]
        # Add some padding around the face
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        face_crop = frame[y1:y2, x1:x2]
        return face_crop
    return None

# Define skeleton connections for hand/finger pose estimation
# Standard 21-point hand keypoint connections (MediaPipe style)
FINGER_CONNECTIONS = [
    # Thumb (keypoints 1-4)
    (0, 1), (1, 2), (2, 3), (3, 4),
    
    # Index finger (keypoints 5-8)
    (0, 5), (5, 6), (6, 7), (7, 8),
    
    # Middle finger (keypoints 9-12)
    (0, 9), (9, 10), (10, 11), (11, 12),
    
    # Ring finger (keypoints 13-16)
    (0, 13), (13, 14), (14, 15), (15, 16),
    
    # Pinky (keypoints 17-20)
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Alternative simplified finger connections if your model has fewer keypoints
SIMPLE_FINGER_CONNECTIONS = [
    # Connect fingertips to palm center (assuming fewer keypoints)
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),  # palm to each fingertip
    # You can adjust these based on your specific model's keypoint layout
]

def draw_pose_skeleton(frame, keypoints, connections=FINGER_CONNECTIONS, conf_threshold=0.5):
    """
    Draw skeleton lines connecting pose keypoints
    
    Args:
        frame: The image frame
        keypoints: Array of keypoints with shape (num_keypoints, 3) where last dim is [x, y, confidence]
        connections: List of tuples defining which keypoints to connect
        conf_threshold: Minimum confidence threshold for drawing connections
    """
    if keypoints is None or len(keypoints) == 0:
        return frame
    
    # Ensure keypoints is the right shape
    if len(keypoints.shape) == 3:
        keypoints = keypoints[0]  # Take first detection if multiple
    
    h, w = frame.shape[:2]
    
    # Draw connections (skeleton)
    for connection in connections:
        kpt1_idx, kpt2_idx = connection
        
        # Check if both keypoints exist and have sufficient confidence
        if (kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints) and
            len(keypoints[kpt1_idx]) >= 3 and len(keypoints[kpt2_idx]) >= 3 and
            keypoints[kpt1_idx][2] > conf_threshold and keypoints[kpt2_idx][2] > conf_threshold):
            
            # Get keypoint coordinates
            x1, y1 = int(keypoints[kpt1_idx][0]), int(keypoints[kpt1_idx][1])
            x2, y2 = int(keypoints[kpt2_idx][0]), int(keypoints[kpt2_idx][1])
            
            # Draw line between keypoints
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw keypoints on top of lines
    for i, kpt in enumerate(keypoints):
        if len(kpt) >= 3 and kpt[2] > conf_threshold:
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            # Optional: add keypoint index
            cv2.putText(frame, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame

# Initialize Streamlit app
st.title("Sign2World")

# Load the models
model = load_model()
face_model, face_model_type = load_face_model()

# Streamlit controls
st.sidebar.title("Controls")
show_face_detection = st.sidebar.checkbox("Enable Face Detection", value=True)
show_pose_estimation = st.sidebar.checkbox("Enable Pose Estimation", value=True)
show_face_crop = st.sidebar.checkbox("Show Face Crop", value=False)

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Unable to access the camera")
else:
    st.write("Camera accessed successfully")

# Streamlit layout
placeholder = st.empty()
face_placeholder = st.empty()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        # Get a copy of the frame for drawing
        annotated_frame = frame.copy()
        faces = []
        
        # Face detection
        if show_face_detection:
            faces = detect_faces(frame, face_model, face_model_type)
            annotated_frame = draw_face_boxes(annotated_frame, faces)
        
        # Pose estimation
        if show_pose_estimation:
            # Perform inference
            results = model(frame)
            result = results[0]
            
            # Extract keypoints if available
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy() if hasattr(result.keypoints, 'conf') else None
                
                # Combine coordinates with confidences
                if confidences is not None:
                    if len(keypoints.shape) == 3:  # Multiple detections
                        for i in range(keypoints.shape[0]):
                            kpts_with_conf = np.concatenate([keypoints[i], confidences[i].reshape(-1, 1)], axis=1)
                            annotated_frame = draw_pose_skeleton(annotated_frame, kpts_with_conf)
                    else:  # Single detection
                        kpts_with_conf = np.concatenate([keypoints, confidences.reshape(-1, 1)], axis=1)
                        annotated_frame = draw_pose_skeleton(annotated_frame, kpts_with_conf)
                else:
                    # If no confidence scores, assume high confidence
                    if len(keypoints.shape) == 3:
                        for i in range(keypoints.shape[0]):
                            kpts_with_conf = np.concatenate([keypoints[i], np.ones((keypoints.shape[1], 1))], axis=1)
                            annotated_frame = draw_pose_skeleton(annotated_frame, kpts_with_conf)
                    else:
                        kpts_with_conf = np.concatenate([keypoints, np.ones((keypoints.shape[0], 1))], axis=1)
                        annotated_frame = draw_pose_skeleton(annotated_frame, kpts_with_conf)
        
        # Convert color space for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_frame)

        # Display the main frame
        placeholder.image(img, caption="Pose Estimation & Face Detection")
        
        # Show face crop if enabled and faces detected
        if show_face_crop and faces:
            face_crop = extract_face_region(frame, faces)
            if face_crop is not None:
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_img = Image.fromarray(face_crop_rgb)
                face_placeholder.image(face_img, caption="Detected Face", width=200)

except Exception as e:
    st.error(f"An error occurred: {e}")

finally:
    cap.release()