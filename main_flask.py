from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import threading
import json
from collections import deque, defaultdict
import base64

app = Flask(__name__)

# === Configuration ===
class Config:
    MODEL_PATH = "yolov5n6.onnx"
    VIDEO_PATH = "video.mp4"  # Bisa diganti ke 0 untuk webcam
    COCO_NAMES_PATH = "coco.names"
    INPUT_SIZE = 320
    CONF_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.4
    TOSCA_COLOR = (208, 224, 64)
    MAX_FPS_HISTORY = 30

# === Global Variables ===
detection_data = {
    'fps': 0,
    'detections': [],
    'frame_count': 0,
    'detection_stats': defaultdict(int),
    'avg_fps': 0,
    'status': 'stopped'
}

fps_history = deque(maxlen=Config.MAX_FPS_HISTORY)
video_capture = None
onnx_session = None
class_names = None
processing_thread = None
is_processing = False

# === Load COCO class names ===
def load_class_names():
    global class_names
    if os.path.exists(Config.COCO_NAMES_PATH):
        with open(Config.COCO_NAMES_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        class_names = [f"Class_{i}" for i in range(80)]  # Default COCO classes

# === Initialize ONNX model ===
def init_onnx_model():
    global onnx_session
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_session = ort.InferenceSession(Config.MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
        return True
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return False

# === Letterbox preprocessing ===
def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img_padded, ratio, (dw, dh)

# === Process frame with YOLO detection ===
def process_frame(frame):
    global onnx_session, class_names, detection_data, fps_history
    
    if onnx_session is None:
        return frame, []
    
    start_time = time.time()
    
    # Preprocessing
    img_input, ratio, (dw, dh) = letterbox(frame, new_shape=(Config.INPUT_SIZE, Config.INPUT_SIZE))
    input_tensor = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output = onnx_session.run(None, {input_name: input_tensor})[0][0]

    # Filter predictions
    conf_mask = output[:, 4] > Config.CONF_THRESHOLD
    output = output[conf_mask]

    detections = []
    
    if output.shape[0] > 0:
        scores = output[:, 4] * output[:, 5:].max(axis=1)
        score_mask = scores > Config.CONF_THRESHOLD
        output = output[score_mask]
        scores = scores[score_mask]
        class_ids = output[:, 5:].argmax(axis=1)

        # Convert coordinates
        cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        cx = (cx - dw) / ratio
        cy = (cy - dh) / ratio
        w = w / ratio
        h = h / ratio

        x = (cx - w / 2).astype(int)
        y = (cy - h / 2).astype(int)
        w = w.astype(int)
        h = h.astype(int)

        boxes = np.stack([x, y, w, h], axis=1)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), Config.CONF_THRESHOLD, Config.NMS_THRESHOLD)

        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            class_id = class_ids[i]
            confidence = scores[i]
            
            label_text = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
            
            # Store detection
            detections.append({
                'class': label_text,
                'confidence': float(confidence),
                'bbox': [int(x), int(y), int(w), int(h)]
            })
            
            # Update stats
            detection_data['detection_stats'][label_text] += 1
            
            # Draw on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), Config.TOSCA_COLOR, 2)
            cv2.putText(frame, f"{label_text}: {confidence:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.TOSCA_COLOR, 2)

    # Calculate FPS
    end_time = time.time()
    elapsed = end_time - start_time
    fps = 1 / elapsed if elapsed > 0 else 0
    fps_history.append(fps)
    
    # Update detection data
    detection_data['fps'] = fps
    detection_data['detections'] = detections
    detection_data['avg_fps'] = sum(fps_history) / len(fps_history) if fps_history else 0
    
    # Add FPS text to frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Detections: {len(detections)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame, detections

# === Video processing thread ===
def video_processing_thread():
    global video_capture, is_processing, detection_data
    
    video_capture = cv2.VideoCapture(Config.VIDEO_PATH)
    if not video_capture.isOpened():
        detection_data['status'] = 'error'
        return
    
    detection_data['status'] = 'running'
    frame_count = 0
    
    while is_processing:
        ret, frame = video_capture.read()
        if not ret:
            # Loop video
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        detection_data['frame_count'] = frame_count
        
        # Process frame
        processed_frame, detections = process_frame(frame)
        
        # Store processed frame for streaming
        detection_data['current_frame'] = processed_frame
        
        time.sleep(0.03)  # Limit to ~30 FPS max
    
    if video_capture:
        video_capture.release()
    detection_data['status'] = 'stopped'

# === Generate video frames for streaming ===
def generate_frames():
    while True:
        if 'current_frame' in detection_data:
            frame = detection_data['current_frame']
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    stats = {
        'fps': round(detection_data['fps'], 1),
        'avg_fps': round(detection_data['avg_fps'], 1),
        'frame_count': detection_data['frame_count'],
        'detections_count': len(detection_data['detections']),
        'detections': detection_data['detections'],
        'detection_stats': dict(detection_data['detection_stats']),
        'status': detection_data['status']
    }
    return jsonify(stats)

@app.route('/api/start')
def start_processing():
    global processing_thread, is_processing
    
    if not is_processing:
        is_processing = True
        processing_thread = threading.Thread(target=video_processing_thread)
        processing_thread.daemon = True
        processing_thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/stop')
def stop_processing():
    global is_processing
    is_processing = False
    return jsonify({'status': 'stopped'})

# === Initialize app ===
def initialize_app():
    load_class_names()
    if not init_onnx_model():
        print("Failed to initialize ONNX model!")
        return False
    return True

if __name__ == '__main__':
    if initialize_app():
        print("YOLOv5 Flask App initialized successfully!")
        print("Starting video processing...")
        
        # Auto-start processing
        is_processing = True
        processing_thread = threading.Thread(target=video_processing_thread)
        processing_thread.daemon = True
        processing_thread.start()
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("Failed to initialize app!")