import socket
import base64
import json
import numpy as np
import cv2
from ultralytics import YOLO
from datetime import datetime
import os

# RPi server IP and port
RPi_HOST = "192.168.11.1"
RPi_PORT = 5005

# YOLO model path
MODEL_PATH = r"G:\NTU\SC2079\MDP_Algo_Ver2\AI\TrainedYOLOnModelColoured.pt"

# Confidence threshold
CONF_THRESHOLD = 0.35

# Directory to save detected frames
SAVE_DIR = r"G:\NTU\SC2079\MDP_Algo_Ver2\AI\saved"
os.makedirs(SAVE_DIR, exist_ok=True)

def receive_full_message(sock):
    """Receive length-prefixed JSON message from RPi"""
    # First read 16-byte length header
    length_bytes = sock.recv(16)
    if not length_bytes:
        return None

    length = int(length_bytes.decode("utf-8").strip())
    data_bytes = b""

    while len(data_bytes) < length:
        chunk = sock.recv(length - len(data_bytes))
        if not chunk:
            break
        data_bytes += chunk

    if len(data_bytes) != length:
        print("Incomplete message received")
        return None

    return json.loads(data_bytes.decode("utf-8"))

def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded.")

    # Connect to RPi
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((RPi_HOST, RPi_PORT))
    print(f"Connected to RPi at {RPi_HOST}:{RPi_PORT}")

    try:
        while True:
            # Receive DETECTION_REQUEST
            message = receive_full_message(sock)
            if not message:
                print("No message received. RPi might have disconnected.")
                break

            if message.get("type") != "DETECTION_REQUEST":
                continue

            obstacle_id = message.get("obstacle_id", 0)
            img_b64 = message.get("image")

            # Decode image
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Fix inverted image (flip vertically)
            frame = cv2.flip(frame, 0)

            # Run YOLO detection
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

            # Pick largest bounding box
            if len(results.boxes) > 0:
                areas = (results.boxes.xyxy[:, 2] - results.boxes.xyxy[:, 0]) * \
                        (results.boxes.xyxy[:, 3] - results.boxes.xyxy[:, 1])
                max_idx = int(areas.argmax())
                class_id = model.names[int(results.boxes.cls[max_idx].item())]

                # Overlay bounding box and class ID on frame
                x1, y1, x2, y2 = map(int, results.boxes.xyxy[max_idx])
                label = f"Obstacle ID: {obstacle_id}, Class ID: {class_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                # Save the annotated frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(SAVE_DIR, f"detection_{obstacle_id}_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"Saved detected frame: {save_path}")

            else:
                class_id = 0  # No detection

            # Prepare response
            response = {
                "type": "TARGET",
                "obstacle_id": obstacle_id,
                "class_id": class_id
            }
            print(response)

            # Convert to JSON bytes with length header
            response_json = json.dumps(response).encode("utf-8")
            print(response_json)
            # Send back to RPi
            sock.sendall(response_json)

            annotated_frame = results.plot()
            cv2.imshow("PC YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User exit requested.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("PC client closed.")

if __name__ == "__main__":
    main()
