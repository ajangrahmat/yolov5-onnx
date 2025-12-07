import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import RPi.GPIO as GPIO
import threading
from gtts import gTTS
from queue import Queue

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

# === TTS Distance Threshold ===
TTS_DISTANCE_THRESHOLD = 75  # cm - Only announce distance when < 75cm

# === Vibration Alert Threshold ===
VIBRATION_DISTANCE_THRESHOLD = 50  # cm - Activate vibration when distance < 50cm

# === TTS Configuration ===
TTS_ENABLED = True
TTS_LANG = 'id'  # Bahasa Indonesia
TTS_AUDIO_FILE = "suara.mp3"
TTS_PLAYER = "mpg321"  # Bisa diganti dengan "aplay" atau "omxplayer"

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

# === TTS Control Variables ===
tts_queue = Queue()
tts_thread = None
tts_running = True
tts_lock = threading.Lock()

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

# === TTS FUNCTIONS ===
def init_tts():
    """Initialize TTS system"""
    if not TTS_ENABLED:
        return False
    
    try:
        # Test if gTTS and audio player are available
        test_tts = gTTS(text="Test", lang=TTS_LANG)
        print(f"✅ gTTS initialized successfully!")
        print(f"   Language: {TTS_LANG}, Player: {TTS_PLAYER}")
        return True
        
    except Exception as e:
        print(f"❌ TTS initialization failed: {e}")
        return False

def tts_worker():
    """TTS worker thread to handle speech queue"""
    global tts_running
    
    while tts_running:
        try:
            if not tts_queue.empty():
                message = tts_queue.get(timeout=1)
                if message:
                    with tts_lock:
                        print(f"🔊 [TTS] Speaking: {message}")
                        
                        # Generate TTS
                        tts = gTTS(text=message, lang=TTS_LANG)
                        tts.save(TTS_AUDIO_FILE)
                        
                        # Play audio
                        os.system(f"{TTS_PLAYER} {TTS_AUDIO_FILE} > /dev/null 2>&1")
                        
                        # Clean up audio file
                        if os.path.exists(TTS_AUDIO_FILE):
                            os.remove(TTS_AUDIO_FILE)
                        
                    tts_queue.task_done()
        except:
            continue

def speak(message, priority=False):
    """Add message to TTS queue"""
    if not TTS_ENABLED:
        return
    
    # Clear queue if priority message
    if priority:
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
                tts_queue.task_done()
            except:
                break
    
    # Add message to queue (avoid duplicates)
    if not tts_queue.full():
        tts_queue.put(message)

def stop_tts():
    """Stop TTS system"""
    global tts_running
    
    tts_running = False
    
    # Clean up audio file
    if os.path.exists(TTS_AUDIO_FILE):
        try:
            os.remove(TTS_AUDIO_FILE)
        except:
            pass

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

# === SEPARATED DETECTION FUNCTIONS WITH TTS ===

def process_object_detection(detected_objects):
    """Process object detection results independently with TTS - ALWAYS announce ALL objects"""
    vibration_needed = False
    
    if not detected_objects:
        print("🔍 [VISION] No obstacles detected")
        # Optional: Announce clear path periodically
        # speak("Path clear")
    else:
        unique_objects = list(set(detected_objects))
        print(f"🎯 [VISION] Objects detected: {', '.join(unique_objects)}")
        vibration_needed = True
        
        # === ALWAYS ANNOUNCE ALL DETECTED OBJECTS WITH HIGH PRIORITY ===
        
        # Create object list for TTS announcement
        object_names = []
        for obj in unique_objects:
            object_names.append(obj.lower())
        
        # Single object announcement
        if len(unique_objects) == 1:
            obj_name = unique_objects[0]
            speak(f"Peringatan! {obj_name} terdeteksi di depan!", priority=True)
            print(f"  ℹ️ [VISION] Single obstacle: {obj_name}")
            
        # Multiple objects announcement - announce ALL objects by name with HIGH PRIORITY
        else:
            if len(unique_objects) == 2:
                obj_list = " dan ".join(object_names)
                speak(f"Peringatan! {obj_list} terdeteksi di depan!", priority=True)
                print(f"  ⚠️ [VISION] Multiple obstacles: {', '.join(unique_objects)}")
                
                # Additional immediate announcement for 2 objects to ensure responsiveness
                time.sleep(0.1)  # Small delay to ensure first announcement starts
                speak(f"Dua rintangan: {obj_list}!", priority=True)
                
            elif len(unique_objects) >= 3:
                # For 3 or more objects, list them all
                if len(unique_objects) == 3:
                    obj_list = f"{object_names[0]}, {object_names[1]}, dan {object_names[2]}"
                else:
                    obj_list = ", ".join(object_names[:-1]) + f", dan {object_names[-1]}"
                
                speak(f"Bahaya! Beberapa rintangan terdeteksi: {obj_list}!", priority=True)
                print(f"  🚨 [VISION] Maximum obstacles detected: {', '.join(unique_objects)}")
        
        # === INDIVIDUAL OBJECT DETECTION LOGGING ===
        for obj in unique_objects:
            if obj == "Batu":
                print("  🪨 [VISION] BATU DETECTED!")
            elif obj == "Gundukan":
                print("  ⛰️ [VISION] GUNDUKAN DETECTED!")
            elif obj == "Tiang":
                print("  🗼 [VISION] TIANG DETECTED!")
            else:
                print(f"  🎯 [VISION] {obj.upper()} DETECTED!")
        
        # === SPECIFIC COMBINATION ALERTS WITH IMMEDIATE PRIORITY ===
        if "Batu" in unique_objects and "Gundukan" in unique_objects:
            print("  ⚠️ [VISION] ALERT: Batu dan Gundukan detected together!")
            speak("Kombinasi batu dan gundukan terdeteksi!", priority=True)
        
        if "Tiang" in unique_objects and ("Batu" in unique_objects or "Gundukan" in unique_objects):
            print("  ⚠️ [VISION] OBSTACLE ALERT: Multiple critical obstacles!")
            speak("Beberapa rintangan kritis terdeteksi!", priority=True)
        
        # All major objects detected - Additional critical alert
        if all(obj in unique_objects for obj in ["Batu", "Gundukan", "Tiang"]):
            print("  🚨 [VISION] ALL MAJOR OBSTACLES DETECTED!")
            speak("Kritis! Semua rintangan utama terdeteksi! Berhenti segera!", priority=True)
    
    return vibration_needed

def process_distance_sensor(distance):
    """Process distance sensor results - TTS ONLY when distance < 75cm"""
    distance_status = "Valid" if DISTANCE_MIN_VALID <= distance <= DISTANCE_MAX_VALID else "Invalid"
    print(f"📏 [SENSOR] Distance: {distance} cm ({distance_status})")
    
    vibration_needed = False
    
    # === DISTANCE-BASED TTS ANNOUNCEMENTS - ONLY < 75cm ===
    if distance < DISTANCE_CRITICAL:
        print(f"  🚨 [SENSOR] CRITICAL: Object extremely close! (<{DISTANCE_CRITICAL}cm)")
        speak(f"Darurat! Objek pada jarak {int(distance)} sentimeter! Berhenti sekarang!", priority=True)
        vibration_needed = True
        
    elif distance < DISTANCE_DANGER:
        print(f"  🛑 [SENSOR] DANGER: Object very close - STOP recommended! (<{DISTANCE_DANGER}cm)")
        speak(f"Bahaya! Objek sangat dekat pada {int(distance)} sentimeter!", priority=True)
        vibration_needed = True
        
    elif distance < VIBRATION_DISTANCE_THRESHOLD:
        print(f"  ⚠️ [SENSOR] VIBRATION ALERT: Object within {VIBRATION_DISTANCE_THRESHOLD}cm!")
        speak(f"Peringatan! Objek pada jarak {int(distance)} sentimeter!")
        vibration_needed = True
        
    elif distance < TTS_DISTANCE_THRESHOLD:  # Only announce if < 75cm
        print(f"  ⚠️ [SENSOR] TTS ALERT: Object within {TTS_DISTANCE_THRESHOLD}cm - announcing distance")
        speak(f"Hati-hati! Objek pada jarak {int(distance)} sentimeter!")
        
    elif distance < DISTANCE_WARNING:
        print(f"  ⚠️ [SENSOR] WARNING: Object nearby - NO TTS (>={TTS_DISTANCE_THRESHOLD}cm)")
        # NO TTS - distance >= 75cm
        
    elif distance < DISTANCE_CAUTION:
        print(f"  ℹ️ [SENSOR] CAUTION: Object detected at moderate distance - NO TTS")
        # NO TTS - distance >= 75cm
        
    else:
        print("  ✅ [SENSOR] CLEAR: Safe distance")
    
    return vibration_needed

def process_temperature_monitoring(suhu):
    """Process temperature monitoring independently with TTS"""
    status_temp = status_suhu(suhu)
    print(f"🌡️ [SYSTEM] CPU Temperature: {suhu:.1f}°C -> {status_temp}")
    
    # === TEMPERATURE-BASED TTS ANNOUNCEMENTS ===
    if suhu > 80:
        print(f"  🚨 [SYSTEM] CRITICAL TEMP: CPU at {suhu:.1f}°C - Consider stopping!")
        speak(f"Peringatan suhu kritis! CPU pada {int(suhu)} derajat!", priority=True)
        
    elif suhu > 70:
        print(f"  🔥 [SYSTEM] HIGH TEMP WARNING: CPU running hot at {suhu:.1f}°C!")
        speak(f"Peringatan suhu tinggi! CPU pada {int(suhu)} derajat!")
        
    elif suhu > 60:
        print(f"  ⚠️ [SYSTEM] Elevated temperature: Monitor closely")
        # Optional: Less frequent temperature announcements
        # speak("Suhu meningkat terdeteksi")

def detect_objects():
    global tts_thread
    
    # Initialize TTS
    tts_available = init_tts()
    
    # Start TTS worker thread
    if TTS_ENABLED and tts_available:
        tts_thread = threading.Thread(target=tts_worker)
        tts_thread.daemon = True
        tts_thread.start()
        
        # Welcome message
        speak("Sistem deteksi dengan panduan suara diaktifkan!")
    
    # Load model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print("=== DETECTION SYSTEM WITH UPDATED TTS RULES ===")
    print("Vision System: Object Detection (ALWAYS announce ALL objects)")
    print(f"Sensor System: Distance Measurement (TTS only when < {TTS_DISTANCE_THRESHOLD}cm)") 
    print("Monitor System: Temperature Monitoring")
    print("Alert Systems: Vibration + TTS Voice Guidance")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            print(f"\n{'='*20} Frame {frame_count} {'='*20}")
            
            # === 1. OBJECT DETECTION (ALWAYS ANNOUNCE ALL OBJECTS) ===
            
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
                
                # Handle different OpenCV versions
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
            
            # Process object detection with TTS (ALWAYS announce ALL objects)
            object_vibration_needed = process_object_detection(detected_objects)
            
            print()  # Separator
            
            # === 2. DISTANCE SENSOR WITH TTS (ONLY < 75cm) ===
            distance = get_distance()
            distance_vibration_needed = process_distance_sensor(distance)
            
            print()  # Separator
            
            # === 3. TEMPERATURE MONITORING WITH TTS ===
            suhu = get_suhu()
            process_temperature_monitoring(suhu)
            
            print()  # Separator
            
            # === 4. VIBRATION & TTS CONTROL ===
            if object_vibration_needed or distance_vibration_needed:
                if object_vibration_needed and distance_vibration_needed:
                    print("  📳 [VIBRATION] DUAL ALERT: Objects detected AND distance < 50cm!")
                    speak("Peringatan ganda! Objek terdeteksi dan jarak terlalu dekat!", priority=True)
                elif object_vibration_needed:
                    print("  📳 [VIBRATION] OBJECT ALERT: Vibrating due to object detection!")
                elif distance_vibration_needed:
                    print("  📳 [VIBRATION] DISTANCE ALERT: Vibrating due to close proximity!")
                
                start_vibration()
            else:
                stop_vibration_alert()
            
            print("-" * 60)
            
            # Small delay to prevent TTS queue overflow but maintain responsiveness
            time.sleep(0.05)  # Reduced from 0.1 to 0.05 for better responsiveness
            
    except KeyboardInterrupt:
        print("\n=== DETECTION SYSTEMS STOPPED ===")
        speak("Sistem deteksi dimatikan", priority=True)
        time.sleep(3)  # Allow final TTS to complete
    
    # Cleanup
    stop_vibration_alert()
    stop_tts()
    cap.release()
    GPIO.cleanup()

if __name__ == "__main__":
    detect_objects()