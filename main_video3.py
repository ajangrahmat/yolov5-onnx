import cv2
import numpy as np
import onnxruntime as ort
import os

# === Configuration ===
model_path = "best_versi_ringan_320.onnx"
video_path = 0 
coco_names_path = "coco.names"
input_size = 320
conf_threshold = 0.25
nms_threshold = 0.4

# === Load class names ===
if os.path.exists(coco_names_path):
    with open(coco_names_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = None

def letterbox(img, new_shape=(320, 320)):
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return img_padded, ratio, (dw, dh)

def detect_objects():
    # Load model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print("Detection started. Press Ctrl+C to stop.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Preprocess
            img_input, ratio, (dw, dh) = letterbox(frame, (input_size, input_size))
            input_tensor = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # Inference
            output = session.run(None, {input_name: input_tensor})[0][0]
            
            # Filter by confidence
            conf_mask = output[:, 4] > conf_threshold
            output = output[conf_mask]
            
            detected_objects = []
            
            if output.shape[0] > 0:
                scores = output[:, 4] * output[:, 5:].max(axis=1)
                score_mask = scores > conf_threshold
                output = output[score_mask]
                scores = scores[score_mask]
                class_ids = output[:, 5:].argmax(axis=1)
                
                # Convert to original coordinates
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
                    class_id = class_ids[i]
                    confidence = scores[i]
                    label = class_names[class_id] if class_names else f"Class {class_id}"
                    detected_objects.append(label)
            
            # Print detected objects and handle conditions
            if detected_objects:
                unique_objects = list(set(detected_objects))
                print(f"Frame {frame_count}: {', '.join(unique_objects)}")
                
                # === IF CONDITIONS FOR DETECTED OBJECTS ===
                
                # Detect specific objects (only 3 classes)
                if "Batu" in unique_objects:
                    print("  🪨 BATU DETECTED!")
                
                if "Gundukan" in unique_objects:
                    print("  ⛰️ GUNDUKAN DETECTED!")
                
                if "Tiang" in unique_objects:
                    print("  🗼 TIANG DETECTED!")
                
                # Multiple object conditions
                if "Batu" in unique_objects and "Gundukan" in unique_objects:
                    print("  ⚠️ ALERT: Batu dan Gundukan terdeteksi bersamaan!")
                
                if "Tiang" in unique_objects and ("Batu" in unique_objects or "Gundukan" in unique_objects):
                    print("  ⚠️ OBSTACLE ALERT: Multiple obstacles detected!")
                
                # All objects detected
                if all(obj in unique_objects for obj in ["Batu", "Gundukan", "Tiang"]):
                    print("  🚨 ALL OBSTACLES DETECTED: Batu, Gundukan, dan Tiang!")
                
                # Count-based conditions
                if len(unique_objects) == 1:
                    print(f"  ℹ️ Single obstacle: {unique_objects[0]}")
                elif len(unique_objects) == 2:
                    print(f"  ⚠️ Double obstacles detected!")
                elif len(unique_objects) == 3:
                    print(f"  🚨 Maximum obstacles detected!")
                
                # Navigation alerts
                obstacles_detected = [obj for obj in unique_objects if obj in ["Batu", "Gundukan", "Tiang"]]
                if obstacles_detected:
                    print(f"  🚧 NAVIGATION WARNING: Avoid {', '.join(obstacles_detected)}")
                
                print()  # Empty line for readability
            
    except KeyboardInterrupt:
        print("\nDetection stopped.")
    
    cap.release()

if __name__ == "__main__":
    detect_objects()