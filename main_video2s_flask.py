from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
import onnxruntime as ort
import time
import threading

app = Flask(__name__)

# === Konfigurasi Model ===
model_path = "best_versi_ringan_416.onnx"
input_size = (416, 416)
conf_threshold = 0.7  # Global variable for confidence threshold
fps = 0.0  # Global variable for FPS
detections = []  # Global variable for detections

# === Inisialisasi Session YOLO ===
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# === Video Capture & Frame Buffer ===
cap = cv2.VideoCapture(4)  # Ganti ke 0 jika ingin webcam default
latest_frame = None
lock = threading.Lock()

# === Resize util ===
def resize_fast(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

# === Proses deteksi berjalan terus di thread terpisah ===
def detection_loop():
    global latest_frame, fps, detections
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_id += 1
        start = time.time()

        img = resize_fast(frame, input_size)
        input_tensor = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = session.run(None, {input_name: input_tensor})[0][0]
        mask = outputs[:, 4] > conf_threshold
        detections = outputs[mask]

        # === TAMPILKAN DEBUG HANYA SETIAP 10 FRAME ===
        if frame_id % 10 == 0:
            print(f"\n[Frame {frame_id}] Deteksi: {len(detections)} objek")

        for i, det in enumerate(detections):
            cx, cy, w, h = det[:4]
            conf = det[4]
            x1 = int((cx - w / 2) * frame.shape[1] / input_size[0])
            y1 = int((cy - h / 2) * frame.shape[0] / input_size[1])
            x2 = int((cx + w / 2) * frame.shape[1] / input_size[0])
            y2 = int((cy + h / 2) * frame.shape[0] / input_size[1])

            if frame_id % 10 == 0:
                print(f"  - Obj {i+1}: Conf={conf:.2f}, Box=({x1},{y1}) - ({x2},{y2})")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        fps = 1 / (time.time() - start + 1e-5)
        if frame_id % 10 == 0:
            print(f"  => Estimasi FPS: {fps:.2f}")

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        with lock:
            latest_frame = frame.copy()

# === Stream Generator untuk Flask ===
def generate_stream():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global fps, detections
    with lock:
        return jsonify({
            "objects": len(detections) if detections is not None else 0,
            "fps": round(fps, 2) if fps is not None else 0.0
        })

@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    global conf_threshold
    data = request.json
    conf_threshold = 0.9 if data.get('high_confidence') else 0.7
    return jsonify({"message": f"Confidence threshold set to {conf_threshold}"})

# === Jalankan Aplikasi ===
if __name__ == '__main__':
    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()
    app.run(debug=True)