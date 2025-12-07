import cv2
import numpy as np
import onnxruntime as ort
import os
import time

# === Path model dan gambar ===
model_path = "yolov5n6.onnx"
image_path = "image.png"
coco_names_path = "coco.names"

# === Parameter deteksi ===
input_size = 320
conf_threshold = 0.25  # dari 0.4 jadi 0.25
nms_threshold = 0.4    # dari 0.5 jadi 0.4

# === Load class labels COCO ===
if os.path.exists(coco_names_path):
    with open(coco_names_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = None

tosca = (208, 224, 64)

def get_color(_): return tosca

# === Fungsi letterbox resize ===
def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    """
    Resize image to a new shape with unchanged aspect ratio using padding
    """
    shape = img.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # resize
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # add border
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img_padded, ratio, (dw, dh)

# === Load gambar ===
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Gagal membaca gambar: {image_path}")
orig_h, orig_w = image.shape[:2]

# === Preprocessing dengan letterbox ===
input_image, ratio, (dw, dh) = letterbox(image, new_shape=(input_size, input_size))

input_tensor = input_image.transpose(2, 0, 1).astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

# === Load model ONNX ===
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# === Mulai timer ===
start_time = time.time()

# === Inference ===
output = session.run(None, {input_name: input_tensor})[0][0]

# === Filter prediksi confidence
conf_mask = output[:, 4] > conf_threshold
output = output[conf_mask]

if output.shape[0] > 0:
    scores = output[:, 4] * output[:, 5:].max(axis=1)
    score_mask = scores > conf_threshold
    output = output[score_mask]
    scores = scores[score_mask]
    class_ids = output[:, 5:].argmax(axis=1)

    # Karena kita pakai letterbox, kita perlu sesuaikan box ke ukuran asli gambar
    # output[:, 0:4] format: [center_x, center_y, width, height] dalam skala input (320x320)
    # kita reverse padding dan scaling:
    cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

    # Sesuaikan koordinat dari input model ke gambar asli:
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

    end_time = time.time()
    print(f"Deteksi selesai dalam {end_time - start_time:.3f} detik.")

    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        label_text = class_names[class_ids[i]] if class_names else f"Class {class_ids[i]}"
        label = f"{label_text}: {scores[i]:.2f}"

        cv2.rectangle(image, (x, y), (x + w, y + h), tosca, 1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tosca, 1)
else:
    end_time = time.time()
    print("Tidak ada objek terdeteksi.")
    print(f"Waktu: {end_time - start_time:.3f} detik.")

cv2.imshow("YOLOv5 Letterbox CPU", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
