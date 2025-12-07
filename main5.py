import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import RPi.GPIO as GPIO
import threading

# === Configuration ===
model_path = "best_versi_ringan_320.onnx"
video_path = 0 
coco_names_path = "coco.names"
input_size = 320
conf_threshold = 0.25
nms_threshold = 0.4

# === Ultrasonic Distance Detection Thresholds ===
DISTANCE_CRITICAL = 10      # cm - Extremely close (CRITICAL)
DISTANCE_DANGER = 30        # cm - Very close (STOP recommended)
DISTANCE_WARNING = 100      # cm - Nearby (Reduce speed)
DISTANCE_CAUTION = 200      # cm - Moderate distance (Be careful)
DISTANCE_MAX_VALID = 400    # cm - Maximum valid reading
DISTANCE_MIN_VALID = 2      # cm - Minimum valid reading

# === Vibration Alert Threshold ===
VIBRATION_DISTANCE_THRESHOLD = 50  # cm - Activate vibration when distance < 50cm

# === GPIO Configuration ===
GPIO.setmode(GPIO.BCM)
TRIG = 14
ECHO = 15
VIBRATOR_PIN = 23  # Pin untuk motor getar

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(VIBRATOR_PIN, GPIO.OUT)

# === Vibration Control Variables ===
vibration_active = False
vibration_thread = None
stop_vibration = False

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

# === Ultrasonic Sensor Functions ===
def get_distance():
    """Get distance from ultrasonic sensor"""
    GPIO.output(TRIG, False)
    time.sleep(0.01)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    pulse_end = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    
    return distance

def get_suhu():
    """Get CPU temperature"""
    try:
        output = os.popen("vcgencmd measure_temp").readline()
        suhu_str = output.replace("temp=", "").replace("'C\n", "")
        return float(suhu_str)
    except:
        return 0.0

def status_suhu(suhu):
    """Classify temperature status"""
    if suhu < 50:
        return "Dingin ❄️"
    elif suhu < 60:
        return "Normal ✅"
    elif suhu < 70:
        return "Agak Panas ⚠️"
    elif suhu < 80:
        return "Panas 🔥"
    else:
        return "Bahaya! 🚨"

# === VIBRATION CONTROL FUNCTIONS ===
def vibration_pattern():
    """Run vibration pattern in separate thread"""
    global stop_vibration
    
    while not stop_vibration:
        # Vibration pattern: 0.5s ON, 0.3s OFF
        GPIO.output(VIBRATOR_PIN, GPIO.HIGH)
        time.sleep(0.5)
        
        if stop_vibration:
            break
            
        GPIO.output(VIBRATOR_PIN, GPIO.LOW)
        time.sleep(0.3)
    
    # Ensure vibrator is off when stopping
    GPIO.output(VIBRATOR_PIN, GPIO.LOW)

def start_vibration():
    """Start vibration alert"""
    global vibration_active, vibration_thread, stop_vibration
    
    if not vibration_active:
        vibration_active = True
        stop_vibration = False
        vibration_thread = threading.Thread(target=vibration_pattern)
        vibration_thread.daemon = True
        vibration_thread.start()
        print("  📳 [VIBRATION] ALERT STARTED!")

def stop_vibration_alert():
    """Stop vibration alert"""
    global vibration_active, stop_vibration
    
    if vibration_active:
        stop_vibration = True
        vibration_active = False
        if vibration_thread and vibration_thread.is_alive():
            vibration_thread.join(timeout=1)
        GPIO.output(VIBRATOR_PIN, GPIO.LOW)
        print("  📳 [VIBRATION] Alert stopped")

# === SEPARATED DETECTION FUNCTIONS ===

def process_object_detection(detected_objects):
    """Process object detection results independently"""
    vibration_needed = False
    
    if not detected_objects:
        print("🔍 [VISION] No obstacles detected")
    else:
        unique_objects = list(set(detected_objects))
        print(f"🎯 [VISION] Objects detected: {', '.join(unique_objects)}")
        vibration_needed = True  # Activate vibration when objects detected
        
        # === OBJECT-ONLY CONDITIONS ===
        
        # Individual object detection
        if "Batu" in unique_objects:
            print("  🪨 [VISION] BATU DETECTED!")
        
        if "Gundukan" in unique_objects:
            print("  ⛰️ [VISION] GUNDUKAN DETECTED!")
        
        if "Tiang" in unique_objects:
            print("  🗼 [VISION] TIANG DETECTED!")
        
        # Multiple object conditions
        if "Batu" in unique_objects and "Gundukan" in unique_objects:
            print("  ⚠️ [VISION] ALERT: Batu dan Gundukan terdeteksi bersamaan!")
        
        if "Tiang" in unique_objects and ("Batu" in unique_objects or "Gundukan" in unique_objects):
            print("  ⚠️ [VISION] OBSTACLE ALERT: Multiple obstacles detected!")
        
        # All objects detected
        if all(obj in unique_objects for obj in ["Batu", "Gundukan", "Tiang"]):
            print("  🚨 [VISION] ALL OBSTACLES DETECTED: Batu, Gundukan, dan Tiang!")
        
        # Count-based conditions
        if len(unique_objects) == 1:
            print(f"  ℹ️ [VISION] Single obstacle: {unique_objects[0]}")
        elif len(unique_objects) == 2:
            print(f"  ⚠️ [VISION] Double obstacles detected!")
        elif len(unique_objects) == 3:
            print(f"  🚨 [VISION] Maximum obstacles detected!")
    
    return vibration_needed

def process_distance_sensor(distance):
    """Process distance sensor results independently"""
    distance_status = "Valid" if DISTANCE_MIN_VALID <= distance <= DISTANCE_MAX_VALID else "Invalid"
    print(f"📏 [SENSOR] Distance: {distance} cm ({distance_status})")
    
    vibration_needed = False
    
    # === DISTANCE-ONLY CONDITIONS ===
    if distance < DISTANCE_CRITICAL:
        print(f"  🚨 [SENSOR] CRITICAL: Object extremely close! (<{DISTANCE_CRITICAL}cm)")
        vibration_needed = True
    elif distance < DISTANCE_DANGER:
        print(f"  🛑 [SENSOR] DANGER: Object very close - STOP recommended! (<{DISTANCE_DANGER}cm)")
        vibration_needed = True
    elif distance < VIBRATION_DISTANCE_THRESHOLD:
        print(f"  ⚠️ [SENSOR] VIBRATION ALERT: Object within {VIBRATION_DISTANCE_THRESHOLD}cm!")
        vibration_needed = True
    elif distance < DISTANCE_WARNING:
        print(f"  ⚠️ [SENSOR] WARNING: Object nearby - Reduce speed! (<{DISTANCE_WARNING}cm)")
    elif distance < DISTANCE_CAUTION:
        print(f"  ℹ️ [SENSOR] CAUTION: Object detected at moderate distance (<{DISTANCE_CAUTION}cm)")
    else:
        print("  ✅ [SENSOR] CLEAR: Safe distance")
    
    return vibration_needed

def process_temperature_monitoring(suhu):
    """Process temperature monitoring independently"""
    status_temp = status_suhu(suhu)
    print(f"🌡️ [SYSTEM] CPU Temperature: {suhu:.1f}°C -> {status_temp}")
    
    # === TEMPERATURE-ONLY CONDITIONS ===
    if suhu > 80:
        print(f"  🚨 [SYSTEM] CRITICAL TEMP: CPU at {suhu:.1f}°C - Consider stopping!")
    elif suhu > 70:
        print(f"  🔥 [SYSTEM] HIGH TEMP WARNING: CPU running hot at {suhu:.1f}°C!")
    elif suhu > 60:
        print(f"  ⚠️ [SYSTEM] Elevated temperature: Monitor closely")

def detect_objects():
    # Load model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print("=== SEPARATED DETECTION SYSTEM WITH VIBRATION ALERT ===")
    print("Vision System: Object Detection")
    print("Sensor System: Distance Measurement") 
    print("Monitor System: Temperature Monitoring")
    print("Vibration System: Alert when distance < 50cm OR objects detected")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            print(f"\n{'='*20} Frame {frame_count} {'='*20}")
            
            # === 1. OBJECT DETECTION (INDEPENDENT) ===
            
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
                
                # Handle different OpenCV versions - indices can be tuple or ndarray
                if isinstance(indices, tuple):
                    indices = indices[0] if len(indices) > 0 else []
                elif isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                else:
                    indices = []
                
                for i in indices:
                    class_id = class_ids[i]
                    confidence = scores[i]
                    label = class_names[class_id] if class_names else f"Class {class_id}"
                    detected_objects.append(label)
            
            # Process object detection results (INDEPENDENT)
            object_vibration_needed = process_object_detection(detected_objects)
            
            print()  # Separator
            
            # === 2. DISTANCE SENSOR (INDEPENDENT) ===
            distance = get_distance()
            distance_vibration_needed = process_distance_sensor(distance)
            
            print()  # Separator
            
            # === 3. TEMPERATURE MONITORING (INDEPENDENT) ===
            suhu = get_suhu()
            process_temperature_monitoring(suhu)
            
            print()  # Separator
            
            # === 4. VIBRATION CONTROL ===
            if object_vibration_needed or distance_vibration_needed:
                if object_vibration_needed and distance_vibration_needed:
                    print("  📳 [VIBRATION] DUAL ALERT: Objects detected AND distance < 50cm!")
                elif object_vibration_needed:
                    print("  📳 [VIBRATION] OBJECT ALERT: Vibrating due to object detection!")
                elif distance_vibration_needed:
                    print("  📳 [VIBRATION] DISTANCE ALERT: Vibrating due to close proximity!")
                
                start_vibration()
            else:
                stop_vibration_alert()
            
            print("-" * 60)
            
    except KeyboardInterrupt:
        print("\n=== DETECTION SYSTEMS STOPPED ===")
    
    # Cleanup
    stop_vibration_alert()
    cap.release()
    GPIO.cleanup()

if __name__ == "__main__":
    detect_objects()