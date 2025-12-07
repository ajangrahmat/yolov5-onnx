import cv2
import numpy as np
import onnxruntime as ort
import os
import time

# === Path model dan video input ===
model_path = "yolov5n6.onnx"   # model yolov5s ONNX
video_path = "video.mp4"
coco_names_path = "coco.names"

# === Parameter deteksi ===
input_size = 320          # input size diubah jadi 640x640
conf_threshold = 0.25
nms_threshold = 0.4

# === Load class labels COCO ===
if os.path.exists(coco_names_path):
    with open(coco_names_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = None

tosca = (208, 224, 64)

def get_color(_): 
    return tosca

# === Fungsi letterbox resize ===
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # height, width
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

# === Load model ONNX ===
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# === Buka video capture ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Gagal membuka video: {video_path}")
    exit()

fps_list = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Preprocessing
    img_input, ratio, (dw, dh) = letterbox(frame, new_shape=(input_size, input_size))
    input_tensor = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Inference
    output = session.run(None, {input_name: input_tensor})[0][0]

    # Filter prediksi confidence
    conf_mask = output[:, 4] > conf_threshold
    output = output[conf_mask]

    if output.shape[0] > 0:
        scores = output[:, 4] * output[:, 5:].max(axis=1)
        score_mask = scores > conf_threshold
        output = output[score_mask]
        scores = scores[score_mask]
        class_ids = output[:, 5:].argmax(axis=1)

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

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, nms_threshold)

        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label_text = class_names[class_ids[i]] if class_names else f"Class {class_ids[i]}"
            label = f"{label_text}: {scores[i]:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), tosca, 1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tosca, 1)

    end_time = time.time()
    elapsed = end_time - start_time
    fps = 1 / elapsed if elapsed > 0 else 0
    fps_list.append(fps)

    cv2.putText(frame, f"FPS: {fps:.2f}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 2)
    cv2.imshow("YOLOv5s ONNX Video Detection", frame)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()

if len(fps_list) > 0:
    avg_fps = sum(fps_list) / len(fps_list)
    print(f"Rata-rata FPS: {avg_fps:.2f}")
