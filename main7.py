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

# === Thresholds ===
DISTANCE_CRITICAL = 10
DISTANCE_DANGER = 30
DISTANCE_WARNING = 75
TEMP_WARNING = 70
TEMP_CRITICAL = 80
VIBRATION_MAX_DURATION = 5.0  # Maksimum durasi getaran (detik)
TTS_HAZARD_INTERVAL = 10.0    # Interval untuk TTS saat kondisi tidak aman (detik)
TTS_SAFE_INTERVAL = 10.0      # Interval untuk TTS "Jalan aman" (detik)

# === GPIO Setup ===
GPIO.setmode(GPIO.BCM)
TRIG, ECHO, VIBRATOR_PIN = 14, 15, 23
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(VIBRATOR_PIN, GPIO.OUT)

# === Global State Tracking ===
current_objects = set()
current_distance_status = None
current_temp_status = None
vibration_active = False
stop_vibration = False
vibration_start_time = 0.0
last_safe_tts_time = 0.0
last_hazard_tts_time = 0.0

# === Sensor Data ===
sensor_data = {
    'distance': 999.0,
    'temperature': 0.0,
    'distance_vibration': False
}

# === System Control ===
system_running = True
tts_queue = Queue(maxsize=5)

# === TTS Cache ===
tts_cache = {}
def cache_tts_messages():
    """Cache common TTS messages as MP3 files"""
    messages = [
        "Sistem deteksi aktif",
        "Jalan aman",
        "Sensor jarak bermasalah",
        "Sistem dimatikan",
        "Area bersih"
    ]
    for msg in messages:
        tts = gTTS(text=msg, lang='id')
        filename = f"tts_cache_{msg.replace(' ', '_')}.mp3"
        tts.save(filename)
        tts_cache[msg] = filename

def speak(message):
    """Play TTS message from cache if available, else generate new"""
    if not tts_queue.full():
        tts_queue.put(message)

def tts_worker():
    """TTS worker with cached audio playback"""
    while system_running:
        try:
            message = tts_queue.get(timeout=1)
            if message:
                if message in tts_cache:
                    os.system(f"mpg321 {tts_cache[message]} > /dev/null 2>&1")
                else:
                    tts = gTTS(text=message, lang='id')
                    temp_file = "temp.mp3"
                    tts.save(temp_file)
                    os.system(f"mpg321 {temp_file} > /dev/null 2>&1")
                    os.remove(temp_file)
                tts_queue.task_done()
        except:
            continue

# === Load class names ===
with open(coco_names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def letterbox(img, new_shape=(320, 320)):
    """Simple letterbox resize"""
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

def get_distance_reading():
    """Get single ultrasonic distance reading with timeout"""
    try:
        # Pastikan TRIG dalam keadaan low sebelum memulai
        GPIO.output(TRIG, False)
        time.sleep(0.0005)  # 500µs delay untuk respons cepat

        # Kirim pulsa 10µs
        GPIO.output(TRIG, True)
        time.sleep(0.00001)  # 10µs pulse
        GPIO.output(TRIG, False)

        # Timeout untuk mencegah loop tak berujung
        timeout = time.time() + 0.1  # Timeout 100ms

        # Tunggu ECHO menjadi HIGH
        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
            if time.time() > timeout:
                print("⚠️ Timeout waiting for ECHO HIGH")
                return 999.0

        # Tunggu ECHO menjadi LOW
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()
            if time.time() > timeout:
                print("⚠️ Timeout waiting for ECHO LOW")
                return 999.0

        # Hitung jarak
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Kecepatan suara: 343 m/s
        distance = round(distance, 2)

        # Validasi jarak (2–400 cm sesuai spesifikasi HC-SR04)
        if distance < 2 or distance > 400:
            print(f"📏 Invalid distance reading: {distance}cm")
            return 999.0

        return distance

    except Exception as e:
        print(f"❌ Ultrasonic error: {e}")
        return 999.0

def get_temperature_reading():
    """Get temperature reading"""
    try:
        output = os.popen("vcgencmd measure_temp").readline()
        suhu_str = output.replace("temp=", "").replace("'C\n", "")
        temp = float(suhu_str)
    except:
        temp = 0.0
    return temp

def vibration_pattern():
    """Simple vibration pattern"""
    global stop_vibration
    while not stop_vibration:
        GPIO.output(VIBRATOR_PIN, GPIO.HIGH)
        time.sleep(0.5)
        if stop_vibration:
            break
        GPIO.output(VIBRATOR_PIN, GPIO.LOW)
        time.sleep(0.3)
    GPIO.output(VIBRATOR_PIN, GPIO.LOW)

def control_vibration(activate):
    """Control vibration on/off with time limit"""
    global vibration_active, stop_vibration, vibration_start_time
    
    if activate and not vibration_active:
        vibration_active = True
        stop_vibration = False
        vibration_start_time = time.time()
        threading.Thread(target=vibration_pattern, daemon=True).start()
        print("📳 Vibration ON")
    elif not activate and vibration_active:
        stop_vibration = True
        vibration_active = False
        print("📳 Vibration OFF")
    elif vibration_active and time.time() - vibration_start_time > VIBRATION_MAX_DURATION:
        stop_vibration = True
        vibration_active = False
        print("📳 Vibration OFF (max duration reached)")

def process_objects(detected_objects):
    """Process object detection with change detection"""
    global current_objects, last_hazard_tts_time
    
    new_objects = set(detected_objects)
    
    # Check if objects changed
    if new_objects != current_objects:
        print(f"🎯 Objects: {list(new_objects) if new_objects else 'None'}")
        
        current_time = time.time()
        if not new_objects:
            speak("Area bersih")
        elif len(new_objects) == 1:
            obj = list(new_objects)[0]
            speak(f"{obj} terdeteksi")
            last_hazard_tts_time = current_time
        elif len(new_objects) == 2:
            objs = list(new_objects)
            speak(f"{objs[0]} dan {objs[1]} terdeteksi")
            last_hazard_tts_time = current_time
        else:
            speak(f"{len(new_objects)} objek terdeteksi")
            last_hazard_tts_time = current_time
        
        current_objects = new_objects
    
    # Continue announcing objects if still detected
    elif new_objects and time.time() - last_hazard_tts_time >= TTS_HAZARD_INTERVAL:
        current_time = time.time()
        if len(new_objects) == 1:
            obj = list(new_objects)[0]
            speak(f"{obj} masih terdeteksi")
        elif len(new_objects) == 2:
            objs = list(new_objects)
            speak(f"{objs[0]} dan {objs[1]} masih terdeteksi")
        else:
            speak(f"{len(new_objects)} objek masih terdeteksi")
        last_hazard_tts_time = current_time
    
    return bool(new_objects)

def detect_objects():
    global system_running, current_distance_status, current_temp_status, last_safe_tts_time, last_hazard_tts_time
    
    # Cache TTS messages
    cache_tts_messages()
    
    # Start TTS thread
    print("🚀 Starting TTS thread...")
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    
    speak("Sistem deteksi aktif!")
    
    # Load model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    cap = cv2.VideoCapture(video_path)
    
    print("=== SEQUENTIAL DETECTION SYSTEM ===")
    print("🎯 Object detection: Main thread (10Hz)")
    print("📏 Ultrasonic sensor: Main thread (2Hz)")
    print("🌡️ Temperature sensor: Main thread (0.5Hz)")
    print("✅ Sequential sensor readings")
    print("Press Ctrl+C to stop.\n")
    
    last_distance_time = 0
    last_temp_time = 0
    
    try:
        while True:
            current_time = time.time()
            
            # === Object Detection ===
            ret, frame = cap.read()
            if not ret:
                break
            
            img_input, ratio, (dw, dh) = letterbox(frame)
            input_tensor = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            output = session.run(None, {input_name: input_tensor})[0][0]
            conf_mask = output[:, 4] > conf_threshold
            output = output[conf_mask]
            
            detected_objects = []
            if output.shape[0] > 0:
                scores = output[:, 4] * output[:, 5:].max(axis=1)
                score_mask = scores > conf_threshold
                output = output[score_mask]
                scores = scores[score_mask]
                class_ids = output[:, 5:].argmax(axis=1)
                
                # Convert coordinates
                cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
                cx = (cx - dw) / ratio
                cy = (cy - dh) / ratio
                w, h = w / ratio, h / ratio
                x, y = (cx - w/2).astype(int), (cy - h/2).astype(int)
                w, h = w.astype(int), h.astype(int)
                
                boxes = np.stack([x, y, w, h], axis=1)
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, nms_threshold)
                
                if isinstance(indices, tuple):
                    indices = indices[0] if len(indices) > 0 else []
                elif isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                
                for i in indices:
                    class_id = class_ids[i]
                    label = class_names[class_id]
                    detected_objects.append(label)
            
            # Process objects
            obj_vibration = process_objects(detected_objects)
            
            # === Ultrasonic Sensor (2Hz) ===
            if current_time - last_distance_time >= 0.5:
                # Ambil 3 pembacaan untuk meningkatkan akurasi
                distances = []
                for _ in range(3):
                    distance = get_distance_reading()
                    if distance < 999.0:  # Hanya gunakan pembacaan valid
                        distances.append(distance)
                    time.sleep(0.01)  # Jeda kecil antar pembacaan
                
                # Gunakan median jika ada pembacaan valid
                if distances:
                    distance = sorted(distances)[len(distances) // 2]
                else:
                    distance = 999.0
                
                # Process distance status
                if distance >= 999.0:
                    status = "error"
                    message = "Sensor jarak bermasalah" if current_distance_status != "error" else None
                    vibration_needed = False
                    if current_distance_status == "error" and time.time() - last_hazard_tts_time >= TTS_HAZARD_INTERVAL:
                        message = "Sensor jarak bermasalah"
                        last_hazard_tts_time = current_time
                elif distance < DISTANCE_CRITICAL:
                    status = "critical"
                    message = f"Darurat! {int(distance)} sentimeter!" if current_distance_status != "critical" else None
                    vibration_needed = True
                    if current_distance_status == "critical" and time.time() - last_hazard_tts_time >= TTS_HAZARD_INTERVAL:
                        message = f"Darurat! {int(distance)} sentimeter masih terdeteksi!"
                        last_hazard_tts_time = current_time
                elif distance < DISTANCE_DANGER:
                    status = "danger"
                    message = f"Bahaya! {int(distance)} sentimeter!" if current_distance_status != "danger" else None
                    vibration_needed = True
                    if current_distance_status == "danger" and time.time() - last_hazard_tts_time >= TTS_HAZARD_INTERVAL:
                        message = f"Bahaya! {int(distance)} sentimeter masih terdeteksi!"
                        last_hazard_tts_time = current_time
                elif distance < DISTANCE_WARNING:
                    status = "warning"
                    message = f"Hati-hati! {int(distance)} sentimeter!" if current_distance_status != "warning" else None
                    vibration_needed = True
                    if current_distance_status == "warning" and time.time() - last_hazard_tts_time >= TTS_HAZARD_INTERVAL:
                        message = f"Hati-hati! {int(distance)} sentimeter masih terdeteksi!"
                        last_hazard_tts_time = current_time
                else:
                    status = "safe"
                    message = None
                    vibration_needed = False
                
                # Update sensor data
                sensor_data['distance'] = distance
                sensor_data['distance_vibration'] = vibration_needed
                
                # Announce status change
                if status != current_distance_status:
                    print(f"📏 Distance: {distance}cm ({status})")
                    if message:
                        speak(message)
                    if status == "safe" and current_distance_status != "safe" and not current_objects:
                        if time.time() - last_safe_tts_time >= TTS_SAFE_INTERVAL:
                            speak("Jalan aman")
                            last_safe_tts_time = current_time
                    current_distance_status = status
                
                last_distance_time = current_time
            
            # === Temperature Sensor (0.5Hz) ===
            if current_time - last_temp_time >= 2.0:
                temp = get_temperature_reading()
                
                # Process temperature status
                if temp > TEMP_CRITICAL:
                    status = "critical"
                    message = f"Suhu kritis! {int(temp)} derajat!" if current_temp_status != "critical" else None
                    if current_temp_status == "critical" and time.time() - last_hazard_tts_time >= TTS_HAZARD_INTERVAL:
                        message = f"Suhu kritis! {int(temp)} derajat masih terdeteksi!"
                        last_hazard_tts_time = current_time
                elif temp > TEMP_WARNING:
                    status = "warning"
                    message = f"Suhu tinggi! {int(temp)} derajat!" if current_temp_status != "warning" else None
                    if current_temp_status == "warning" and time.time() - last_hazard_tts_time >= TTS_HAZARD_INTERVAL:
                        message = f"Suhu tinggi! {int(temp)} derajat masih terdeteksi!"
                        last_hazard_tts_time = current_time
                else:
                    status = "normal"
                    message = None
                
                # Update sensor data
                sensor_data['temperature'] = temp
                
                # Announce status change
                if status != current_temp_status:
                    print(f"🌡️ Temperature: {temp:.1f}°C ({status})")
                    if message:
                        speak(message)
                    current_temp_status = status
                
                last_temp_time = current_time
            
            # Control vibration with time limit
            control_vibration(obj_vibration or sensor_data['distance_vibration'])
            
            time.sleep(0.1)  # 10Hz main loop
            
    except KeyboardInterrupt:
        print("\n=== STOPPING SYSTEM ===")
        speak("Sistem dimatikan")
        time.sleep(2)
    
    # Cleanup
    system_running = False
    control_vibration(False)
    cap.release()
    GPIO.cleanup()
    # Remove cached TTS files
    for filename in tts_cache.values():
        if os.path.exists(filename):
            os.remove(filename)
    
    print("✅ System stopped, GPIO cleaned up")

if __name__ == "__main__":
    detect_objects()