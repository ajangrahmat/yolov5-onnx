import cv2
import numpy as np
import onnxruntime as ort
import os
import time

# === Path model dan video input ===
model_path = "best_versi_ringan_416.onnx"      
video_path = "video.mp4"                       
coco_names_path = "coco2.names"       

# === Path output video ===
output_video_path = "output_detection.mp4"

# === Parameter deteksi ===
input_size = (416, 416)  # Tinggi x Lebar
# conf_threshold = 0.25
# nms_threshold = 0.4
conf_threshold = 0.7
nms_threshold = 0.5

# === Load class labels ===
if os.path.exists(coco_names_path):
    with open(coco_names_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = None

# === Pre-generate colors untuk tiap class ID ===
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]

def get_color(class_id):
    return colors[class_id % len(colors)]

# === Optimized Letterbox Resize ===
def letterbox_optimized(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))
    dw = (new_shape[1] - new_unpad[0]) // 2
    dh = (new_shape[0] - new_unpad[1]) // 2
    
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(img_resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img_padded, ratio, (dw, dh)

# === Load model ONNX dengan optimasi ===
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.enable_mem_pattern = False
so.enable_cpu_mem_arena = False
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Gunakan provider yang tersedia dengan urutan prioritas
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = ort.get_available_providers()
session_providers = [p for p in providers if p in available_providers]

session = ort.InferenceSession(model_path, sess_options=so, providers=session_providers)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Model input shape: {input_shape}")
print(f"Using providers: {session.get_providers()}")

# === Buka video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Gagal membuka video: {video_path}")
    exit()

# Optimasi buffer video
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

fps_original = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps_original, (width, height))
if not out.isOpened():
    print("Gagal membuat video output")
    cap.release()
    exit()

fps_list = []
frame_count = 0
print(f"Deteksi dimulai, output: {output_video_path}")

# Pre-allocate arrays untuk mengurangi memory allocation
input_tensor_shape = (1, 3, input_size[0], input_size[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    start_time = time.time()

    # Preprocessing yang dioptimasi
    img_input, ratio, (dw, dh) = letterbox_optimized(frame, new_shape=input_size)
    
    # Optimasi transpose dan normalisasi dalam satu operasi
    input_tensor = img_input.astype(np.float32, copy=False)
    input_tensor = np.transpose(input_tensor, (2, 0, 1)) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Inference
    output = session.run(None, {input_name: input_tensor})[0][0]
    
    # Optimasi filtering menggunakan vectorized operations
    conf_mask = output[:, 4] > conf_threshold
    if not conf_mask.any():
        # Skip jika tidak ada deteksi
        elapsed = time.time() - start_time
        fps = 1 / elapsed if elapsed > 0 else 0
        fps_list.append(fps)
        
        cv2.putText(frame, f"FPS: {fps:.2f}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: 0", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)
        
        out.write(frame)
        cv2.imshow("YOLOv5 ONNX Detection", frame)
        
        if frame_count % 30 == 0:
            print(f"Progress: {(frame_count / total_frames) * 100:.1f}%")
        
        if cv2.waitKey(1) & 0xFF == 27:
            print("Proses dihentikan oleh user.")
            break
        continue

    output = output[conf_mask]
    detection_count = 0
    
    if output.shape[0] > 0:
        # Vectorized score calculation
        class_scores = output[:, 5:]
        max_class_scores = np.max(class_scores, axis=1)
        scores = output[:, 4] * max_class_scores
        
        score_mask = scores > conf_threshold
        if score_mask.any():
            output = output[score_mask]
            scores = scores[score_mask]
            class_ids = np.argmax(class_scores[score_mask], axis=1)

            # Vectorized coordinate transformation
            cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
            cx = (cx - dw) / ratio
            cy = (cy - dh) / ratio
            w /= ratio
            h /= ratio

            # Area filtering
            areas = w * h
            area_mask = areas > 500

            if area_mask.any():
                cx, cy, w, h = cx[area_mask], cy[area_mask], w[area_mask], h[area_mask]
                scores, class_ids = scores[area_mask], class_ids[area_mask]

                # Convert to integers untuk drawing
                x = np.clip((cx - w / 2).astype(np.int32), 0, width)
                y = np.clip((cy - h / 2).astype(np.int32), 0, height)
                w_int = np.clip(w.astype(np.int32), 1, width - x)
                h_int = np.clip(h.astype(np.int32), 1, height - y)

                boxes = np.column_stack([x, y, w_int, h_int])
                
                # NMS
                if len(boxes) > 0:
                    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, nms_threshold)
                    
                    if len(indices) > 0:
                        indices = indices.flatten()
                        detection_count = len(indices)
                        
                        # Batch drawing untuk efisiensi
                        for i in indices:
                            x_i, y_i, w_i, h_i = boxes[i]
                            class_id = class_ids[i]
                            color = get_color(class_id)
                            
                            # Drawing optimized
                            cv2.rectangle(frame, (x_i, y_i), (x_i + w_i, y_i + h_i), color, 2)
                            
                            label_text = class_names[class_id] if class_names else f"Class {class_id}"
                            label = f"{label_text}: {scores[i]:.2f}"
                            
                            # Text background
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(frame, (x_i, y_i - text_h - 10), (x_i + text_w, y_i), color, -1)
                            cv2.putText(frame, label, (x_i, y_i - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    elapsed = time.time() - start_time
    fps = 1 / elapsed if elapsed > 0 else 0
    fps_list.append(fps)

    # Info overlay
    cv2.putText(frame, f"FPS: {fps:.2f}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Objects: {detection_count}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)

    out.write(frame)
    cv2.imshow("YOLOv5 ONNX Detection", frame)

    if frame_count % 30 == 0:
        print(f"Progress: {(frame_count / total_frames) * 100:.1f}%")

    if cv2.waitKey(1) & 0xFF == 27:
        print("Proses dihentikan oleh user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Statistik akhir
if fps_list:
    avg_fps = sum(fps_list) / len(fps_list)
    print("\n=== HASIL PROSES ===")
    print(f"Output: {output_video_path}")
    print(f"Total frame: {frame_count}")
    print(f"Rata-rata FPS: {avg_fps:.2f}")
    if os.path.exists(output_video_path):
        size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"Ukuran file output: {size_mb:.1f} MB")
    else:
        print("PERINGATAN: File output tidak ditemukan!")
else:
    print("Tidak ada frame yang diproses.")