import numpy as np
import threading
import requests
import time
import queue
import cv2
import json
import os
import glob
from flask import Flask, request
from ultralytics import YOLO

# Configurazione

# POST_URL = "http://192.168.1.159:30033/api/robotic_orchestrator/submit_task"  # production URL
POST_URL = "http://localhost:8000/test"  # Local testing endpoint
YOLO_CONFIDENCE_THRESHOLD = 0.6
DETECTIONS_THRESHOLD = 5
SPOT_SERVER_HOST = "0.0.0.0"
SPOT_SERVER_PORT = 5000
MODEL_PATH = '/home/enf/Downloads/1m/detector_v8.pt'
TEST_IMAGES_PATH = '/home/enf/Downloads/1m/*.jpg'


# Altre variabili

STOP_EVENT = threading.Event()
IMAGE_QUEUE = queue.Queue()
CONSECUTIVE_DETECTIONS = 0
CONSECUTIVE_DETECTIONS_LOCK = threading.Lock()

# Payload POST

delivery_payload_template = [
    {
        "category": "delivery",
        "task": {
            "pickup": [
                {
                    "place": "ur_load",
                    "payload": [
                        {
                            "sku": "pliers",
                            "quantity": 1
                        },
                        {
                            "sku": "fence",
                            "quantity": 1
                        },
                        {
                            "sku": "pincher",
                            "quantity": 1
                        }
                    ]
                }
            ],
            "dropoff": [  # Qui inseriremo dinamicamente il valore passato
                {
                    "place": "p2",
                    "payload": [
                        {
                            "sku": "pliers",
                            "quantity": 1
                        },
                        {
                            "sku": "fence",
                            "quantity": 1
                        },
                        {
                            "sku": "pincher",
                            "quantity": 1
                        }
                    ]
                }
            ],
            "millis": 1
        }
    }
]


app = Flask(__name__)

# COMMENTED OUT FOR LOCAL TESTING - FLASK SERVER
@app.route('/spot/image', methods=['POST'])
def receive_spot_image():
    """
    Endpoint to receive images from SPOT robot via POST request.
    Expects image data in the request.
    """
    try:
        # Get image data from POST request
        if 'image' in request.files:
            # If image is sent as file
            image_file = request.files['image']
            image_data = image_file.read()
        elif request.data:
            # If image is sent as raw data
            image_data = request.data
        else:
            return {"error": "No image data found"}, 400
        
        # Convert to numpy array (RGB format)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB (OpenCV loads as BGR by default)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"[RECEIVER] Image received from SPOT. Shape: {image_rgb.shape}")
            IMAGE_QUEUE.put(image_rgb)
            return {"status": "success", "message": "Image received"}, 200
        else:
            return {"error": "Failed to decode image"}, 400
            
    except Exception as e:
        print(f"[RECEIVER] Error receiving image: {e}")
        return {"error": str(e)}, 500

COMMENTED OUT FOR LOCAL TESTING - FLASK SERVER
def receive_image_from_spot():
    """
    Runs Flask server to receive images from SPOT robot via POST requests.
    Images are expected in RGB format and placed into IMAGE_QUEUE.
    """
    print(f"[RECEIVER] Starting SPOT image server on {SPOT_SERVER_HOST}:{SPOT_SERVER_PORT}")
    app.run(host=SPOT_SERVER_HOST, port=SPOT_SERVER_PORT, debug=False, threaded=True)
        

# FOR TESTING PURPOSES - LOAD LOCAL IMAGES
def receive_image_from_spot():
    """
    Simulates receiving images from SPOT robot by loading local test images.
    Loads images one by one from the test directory.
    """
    print(f"[RECEIVER] Loading test images from {TEST_IMAGES_PATH}")
    
    # Get list of all JPG images
    image_files = glob.glob(TEST_IMAGES_PATH)
    image_files.sort()  # Sort for consistent order
    
    if not image_files:
        print(f"[RECEIVER] ❌ No images found in {TEST_IMAGES_PATH}")
        return
    
    print(f"[RECEIVER] Found {len(image_files)} test images")
    
    image_index = 0
    while not STOP_EVENT.is_set():
        if image_index >= len(image_files):
            image_index = 0  # Loop back to first image
            print("[RECEIVER] 🔄 Restarting image sequence...")
        
        image_path = image_files[image_index]
        image_name = os.path.basename(image_path)
        
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is not None:
                # Convert BGR to RGB (OpenCV loads as BGR by default)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(f"[RECEIVER] 📸 Loaded: {image_name} - Shape: {image_rgb.shape}")
                IMAGE_QUEUE.put(image_rgb)
            else:
                print(f"[RECEIVER] ❌ Failed to load: {image_name}")
        except Exception as e:
            print(f"[RECEIVER] ❌ Error loading {image_name}: {e}")
        
        image_index += 1
        time.sleep(3)  # Wait 3 seconds between images
        

# DON'T DELETE THIS IT'S FOR TESTING PURPOSES
# def receive_image_from_spot():
#     """
#     Simulates receiving an image from the SPOT robot.
#     Replace this with your actual image receiving logic.
#     """
#     while not STOP_EVENT.is_set():
#         # Simulate image arrival
#         print("[RECEIVER] Waiting for image...")
#         time.sleep(2)  # Simulate time delay from robot
#         dummy_image = "image_data"
#         print("[RECEIVER] Image received.")
#         IMAGE_QUEUE.put(dummy_image)


def run_inference(image):
    """
    Run YOLO inference on the image using Ultralytics YOLO.
    Uses a 5-frame consecutive detection heuristic to avoid false positives.
    Return True if detection is positive for 5 consecutive frames.
    """
    global CONSECUTIVE_DETECTIONS

    print("[INFERENCE] Running inference...")
    
    try:
        # Load YOLO model (ideally do this once outside function for better performance)
        model = YOLO(MODEL_PATH)
        
        # Run inference on the image
        results = model(image, verbose=False)  # verbose=False to reduce console output
        
        # Process results
        detected = False
        confidence = 0.0
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Get the highest confidence detection
                confidences = result.boxes.conf.cpu().numpy()  # Convert to numpy array
                max_conf = float(np.max(confidences))
                
                print(f"\n\n\n{max_conf = }\n\n\n")
                
                if max_conf >= YOLO_CONFIDENCE_THRESHOLD:
                    detected = True
                    confidence = max_conf
                    break

        print(f"[INFERENCE] Detection: {detected}, Confidence: {confidence:.2f}")

        with CONSECUTIVE_DETECTIONS_LOCK:
            if detected and confidence >= YOLO_CONFIDENCE_THRESHOLD:
                CONSECUTIVE_DETECTIONS += 1
                print(f"[INFERENCE] Consecutive detections: {CONSECUTIVE_DETECTIONS}/{DETECTIONS_THRESHOLD}")

                if CONSECUTIVE_DETECTIONS >= DETECTIONS_THRESHOLD:
                    CONSECUTIVE_DETECTIONS = 0  # Reset counter
                    return True
            else:
                CONSECUTIVE_DETECTIONS = 0  # Reset on failed detection

        return False
        
    except Exception as e:
        print(f"[INFERENCE] Error during inference: {e}")
        # Reset counter on error
        with CONSECUTIVE_DETECTIONS_LOCK:
            CONSECUTIVE_DETECTIONS = 0
        return False


def send_alert():
    """
    Send POST request to backend.
    Replace payload with your actual format.
    """
    payload = {
        "message": "Positive detection from SPOT robot.",
        "timestamp": time.time()
    }
    try:
        response = requests.post(POST_URL, json=payload)
        print(f"[POST] Sent alert. Status: {response.status_code}")
    except Exception as e:
        print(f"[POST] Failed to send alert: {e}")


def inference_worker():
    """
    Thread that processes images from the queue and runs inference.
    Sends POST if detection is positive.
    """
    while not STOP_EVENT.is_set():
        try:
            image = IMAGE_QUEUE.get(timeout=1)
        except queue.Empty:
            continue
        if run_inference(image):
            print("[INFERENCE] Positive detection.")
            send_alert()
        else:
            print("[INFERENCE] No relevant detection.")
            
        IMAGE_QUEUE.task_done()


# === HEALTH CHECKS ===

def check_backend_connection():
    """
    Check if the backend server is reachable.
    Returns True if connection is successful, False otherwise.
    """
    try:
        print(f"[HEALTH CHECK] Testing backend connection to {POST_URL}...")
        
        # Try a simple GET request first (less intrusive than POST)
        base_url = POST_URL.rsplit('/', 1)[0]  # Remove endpoint path
        response = requests.get(base_url, timeout=5)
        
        if response.status_code in [200, 404]:  # 404 is OK, means server is running
            print(f"[HEALTH CHECK] ✓ Backend server is reachable")
            return True
        else:
            print(f"[HEALTH CHECK] ✗ Backend server responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"[HEALTH CHECK] ✗ Cannot connect to backend server at {POST_URL}")
        return False
    except requests.exceptions.Timeout:
        print(f"[HEALTH CHECK] ✗ Backend server connection timeout")
        return False
    except Exception as e:
        print(f"[HEALTH CHECK] ✗ Backend health check failed: {e}")
        return False


def check_spot_server_port():
    """
    Check if the SPOT server port is available.
    Returns True if port is available, False if occupied.
    """
    import socket
    
    try:
        print(f"[HEALTH CHECK] Checking if port {SPOT_SERVER_PORT} is available...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((SPOT_SERVER_HOST, SPOT_SERVER_PORT))
        sock.close()
        
        if result == 0:
            print(f"[HEALTH CHECK] ✗ Port {SPOT_SERVER_PORT} is already in use")
            return False
        else:
            print(f"[HEALTH CHECK] ✓ Port {SPOT_SERVER_PORT} is available")
            return True
            
    except Exception as e:
        print(f"[HEALTH CHECK] ✗ Port check failed: {e}")
        return False


def wait_for_services():
    """
    Wait for required services to become available.
    Returns True when all services are ready, False if user cancels.
    """
    print("\n" + "="*60)
    print("PERFORMING STARTUP HEALTH CHECKS")
    print("="*60)
    
    max_retries = 30  # 30 attempts
    retry_delay = 2   # 2 seconds between attempts
    
    for attempt in range(1, max_retries + 1):
        print(f"\n[ATTEMPT {attempt}/{max_retries}]")
        
        # Check if SPOT server port is available
        port_available = check_spot_server_port()
        
        # Check backend connection
        backend_ready = check_backend_connection()
        
        if port_available and backend_ready:
            print("\n" + "="*60)
            print("✓ ALL HEALTH CHECKS PASSED - STARTING APPLICATION")
            print("="*60)
            return True
        
        if attempt < max_retries:
            print(f"\n⏳ Waiting {retry_delay} seconds before retry...")
            print("💡 TIP: Make sure your backend server is running!")
            print("💡 TIP: If testing locally, run: python test_server.py")
            time.sleep(retry_delay)
        else:
            print("\n" + "="*60)
            print("✗ HEALTH CHECKS FAILED - MAX RETRIES REACHED")
            print("="*60)
            print("Please check:")
            print(f"1. Backend server is running at {POST_URL}")
            print(f"2. Port {SPOT_SERVER_PORT} is not occupied by another process")
            return False
    
    return False


def main():
    print("="*60)
    print("🧪 LOCAL TESTING MODE - Loading images from disk")
    print("="*60)
    
    # COMMENTED OUT FOR LOCAL TESTING - HEALTH CHECKS
    # if not wait_for_services():
    #     print("❌ Startup failed - exiting application")
    #     return
    
    # Start threads only after health checks pass
    receiver_thread = threading.Thread(
        target=receive_image_from_spot, daemon=False)
    inference_thread = threading.Thread(target=inference_worker, daemon=False)

    receiver_thread.start()
    inference_thread.start()
    
    print(f"📂 Loading test images from: {TEST_IMAGES_PATH}")
    print(f"🎯 Backend alerts will be sent to: {POST_URL}")
    print(f"🔍 YOLO model: {MODEL_PATH}")
    print(f"📊 Detection threshold: {DETECTIONS_THRESHOLD} consecutive detections")
    print(f"🎚️ Confidence threshold: {YOLO_CONFIDENCE_THRESHOLD}")
    print("\nPress Ctrl+C to stop the application\n")
    print("="*60)
    print("👌🏻👌🏻👌🏻👌🏻👌🏻 TUTTO PRONTO!")
    print("="*60)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        STOP_EVENT.set()
        print("⏳ Waiting for threads to finish...")
        receiver_thread.join()
        inference_thread.join()
        print("✅ Fine!")


if __name__ == "__main__":
    main()
