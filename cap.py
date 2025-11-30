
import os
from pathlib import Path

import cv2
import time
import numpy as np
from multiprocessing import Process, Queue
from picamera2 import Picamera2
from filterpy.kalman import KalmanFilter
import serial
import subprocess
import sys

# =========================
# Kalman Tracker (2D center)
# =========================
class KalmanTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0., 0., 0., 0.], dtype=float)  # [x, y, vx, vy]
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.kf.P *= 1000.  # large initial uncertainty
        self.kf.R *= 5.  # measurement noise
        self.kf.Q = np.eye(4) * 0.01  # process noise

    def update(self, cx, cy):
        self.kf.predict()
        self.kf.update([float(cx), float(cy)])
        return float(self.kf.x[0]), float(self.kf.x[1])

# =========================
# Helpers - YuNet Face Detection (OpenCV DNN)
# =========================
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = Path(__file__).resolve().parent / "models" / YUNET_MODEL_FILENAME

def create_face_detector(initial_size=(640, 480)):
    """Create YuNet detector if model file exists."""
    if not YUNET_MODEL_PATH.exists():
        print(
            f"❌ [Arducam] YuNet model not found at {YUNET_MODEL_PATH}. "
            "Download face_detection_yunet_2023mar.onnx from OpenCV Zoo "
            "(https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) "
            "and place it in the models/ directory."
        )
        return None

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        initial_size,
        score_threshold=0.3,
        nms_threshold=0.3,
        top_k=5000
    )
    return detector

def detect_face_center_yunet(frame, detector):
    """Detect face using YuNet and return center + bbox"""
    if detector is None:
        return None

    h, w, _ = frame.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None

    # Use most confident detection (faces[:,4] is score)
    best_idx = int(np.argmax(faces[:, 4]))
    x, y, box_w, box_h, score = faces[best_idx, :5]

    x = int(x)
    y = int(y)
    box_w = int(box_w)
    box_h = int(box_h)
    x_min = max(0, x)
    y_min = max(0, y)
    x_max = min(w - 1, x + box_w)
    y_max = min(h - 1, y + box_h)
    cx = x_min + (x_max - x_min) // 2
    cy = y_min + (y_max - y_min) // 2

    return cx, cy, x_min, y_min, x_max, y_max

def compute_box_center(frame, detector):
    """Return (cx, cy, x_min, y_min, x_max, y_max) using YuNet detection"""
    return detect_face_center_yunet(frame, detector)

def arducam_process(target_queue: Queue):
    """
    Arducam: Picamera2 with native continuous autofocus.
    Runs detection model and sends commands to Arduino.
    """
    import time, cv2, numpy as np, os

    cam_index = 1
    window_name = "Arducam View"

    kf = KalmanTracker()
    DISP_W, DISP_H = 640, 360  # preview size

    # --- Initialize YuNet Face Detector ---
    face_detector = create_face_detector(initial_size=(640, 480))
    if face_detector is None:
        print("❌ [Arducam] Unable to continue without YuNet model.")
        return

    # --- Serial setup for Arduino ---
    try:
        ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        time.sleep(1)
        print("✅ [Arducam] Serial opened")
    except Exception as e:
        print(f"⚠️ [Arducam] Serial unavailable: {e}")
        ser = None

    # --- Control parameters ---
    BASE_SIGN = 1
    WHEEL_SIGN = 1
    DEAD_BAND_NORM = 0.015
    KP_BASE = 34
    KP_WHEEL = 18
    MAX_STEP_BASE = 14
    MAX_STEP_WHEEL = 8
    SEND_PERIOD = 0.03   # send twice as often for tighter control
    EMA_ALPHA = 0.45     # faster exponential smoothing to reduce lag
    LATENCY_LOG_EVERY = 60
    err_x_ema, err_y_ema = 0.0, 0.0
    last_send = 0.0
    last_loop_ts = time.time()
    latency_accum = 0.0
    latency_count = 0

    def send_delta(b_deg, w_deg):
        nonlocal last_send
        if not ser:
            return
        now = time.time()
        if now - last_send < SEND_PERIOD:
            return
        b = int(np.clip(b_deg, -MAX_STEP_BASE, MAX_STEP_BASE))
        w = int(np.clip(w_deg, -MAX_STEP_WHEEL, MAX_STEP_WHEEL))
        if b == 0 and w == 0:
            return
        cmd = f"B{b} W{w}\n"
        try:
            print(f"➡️ [Arducam->Arduino] {cmd.strip()}")
            ser.write(cmd.encode('utf-8'))
        except Exception as e:
            print(f"⚠️ [Arducam] Serial write failed: {e}")
        last_send = now

    # --- Camera init with continuous AF ---
    try:
        camera_info = Picamera2.global_camera_info()
        if not camera_info:
            print("❌ [Arducam] No cameras detected.")
            return
        if cam_index >= len(camera_info):
            print(f"⚠️ Camera index {cam_index} not found. Using index 0.")
            cam_index = 0
    except Exception as e:
        print("❌ Camera info error:", e)
        return

    try:
        picam2 = Picamera2(camera_num=cam_index)
        # Configure camera
        arducam_config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        picam2.configure(arducam_config)
        picam2.start()
        
        # Autofocus for Arducam
        picam2.set_controls({"AfMode": 2})
        picam2.set_controls({"AfTrigger": 0})
        
        time.sleep(1.0)  # Let AF initialize
        print("✅ Arducam: Autofocus enabled via Picamera2.")
    except Exception as e:
        print("❌ Camera init failed:", e)
        return

    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                continue

            h, w, _ = frame.shape

            # Run YuNet face detection on frame
            info = compute_box_center(frame, face_detector)

            # Prepare display frame
            disp = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)

            have_target = False
            if info is not None:
                cx, cy, x_min, y_min, x_max, y_max = info

                # Kalman smoothing
                cx_f, cy_f = kf.update(cx, cy)
                have_target = True

                # Draw on DISPLAY frame (not full-res) so boxes are visible
                # Map detection coords to display coords
                disp_sx = float(DISP_W) / float(w)
                disp_sy = float(DISP_H) / float(h)
                x1_disp = int(x_min * disp_sx)
                x2_disp = int(x_max * disp_sx)
                y1_disp = int(y_min * disp_sy)
                y2_disp = int(y_max * disp_sy)
                cx_disp = int(cx_f * disp_sx)
                cy_disp = int(cy_f * disp_sy)

                # Draw thick bounding box and center on display frame
                cv2.rectangle(disp, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 255, 0), 3)
                cv2.circle(disp, (cx_disp, cy_disp), 6, (0, 0, 255), -1)

                # Send normalized target to queue (for ZoomCam display hint)
                nx = np.clip(cx_f / w, 0.0, 1.0)
                ny = np.clip(cy_f / h, 0.0, 1.0)
                try:
                    while not target_queue.empty():
                        target_queue.get_nowait()
                except Exception:
                    pass
                target_queue.put((nx, ny))

                # --- Send commands to Arduino ---
                now = time.time()
                loop_dt = now - last_loop_ts
                last_loop_ts = now
                latency_accum += loop_dt
                latency_count += 1
                if latency_count >= LATENCY_LOG_EVERY:
                    avg_ms = (latency_accum / latency_count) * 1000.0
                    print(f"⏱️ [Arducam] Avg loop dt: {avg_ms:.1f} ms over {latency_count} frames")
                    latency_accum = 0.0
                    latency_count = 0

                err_x = (cx_f - w / 2) / (w / 2)
                err_y = (cy_f - h / 2) / (h / 2)
                err_x_ema = (1 - EMA_ALPHA) * err_x_ema + EMA_ALPHA * err_x
                err_y_ema = (1 - EMA_ALPHA) * err_y_ema + EMA_ALPHA * err_y
                if abs(err_x_ema) < DEAD_BAND_NORM:
                    err_x_ema = 0.0
                if abs(err_y_ema) < DEAD_BAND_NORM:
                    err_y_ema = 0.0
                move_base = BASE_SIGN * (KP_BASE * err_x_ema)
                move_wheel = WHEEL_SIGN * (-KP_WHEEL * err_y_ema)
                send_delta(move_base, move_wheel)

            # Lightweight preview with overlay
            cv2.putText(disp, "YuNet Face Detection", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(disp, f"Cap: {w}x{h} Disp: {DISP_W}x{DISP_H}", (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow(window_name, disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        try:
            picam2.close()
        except Exception:
            pass
        if ser:
            try:
                ser.close()
            except Exception:
                pass
        cv2.destroyWindow(window_name)



# =========================
# ZoomCam process (Display only - no detection or Arduino commands)
# =========================
def zoomcam_process(target_queue: Queue):
    """
    ZoomCam: Only displays the video feed from UDP stream.
    No detection model, no Arduino commands.
    """
    import subprocess, cv2, time, numpy as np, os
    import threading, select
    from queue import Queue as ThreadQueue, Empty

    window_name = "ZoomCam View"

    # Adjust these for tradeoff between quality and CPU
    OUT_WIDTH, OUT_HEIGHT = 640, 360     # ffmpeg output resolution
    OUT_FPS = 15                         # lower fps reduces CPU
    JPEG_QUEUE_MAX = 2

    # FFmpeg: request a smaller MJPEG stream (less CPU + bandwidth)
    ffmpeg_cmd = [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-i", "udp://@:5000",
        "-an",
        "-c:v", "mjpeg",
        "-q:v", "8",
        "-vf", f"scale={OUT_WIDTH}:{OUT_HEIGHT}",
        "-r", str(OUT_FPS),
        "-f", "image2pipe",
        "-"
    ]

    try:
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**6
        )
        if ffmpeg_proc.stdout is None:
            raise RuntimeError("ffmpeg stdout pipe not available")
        try:
            os.set_blocking(ffmpeg_proc.stdout.fileno(), False)
        except Exception:
            pass
        print("✅ [ZoomCam] FFmpeg receiver started (udp://@:5000).")
    except Exception as e:
        print(f"❌ [ZoomCam] Failed to start FFmpeg: {e}")
        return

    # threaded queues
    jpeg_queue = ThreadQueue(maxsize=JPEG_QUEUE_MAX)
    frame_queue = ThreadQueue(maxsize=JPEG_QUEUE_MAX)

    stop_event = threading.Event()

    # Reader thread: read ffmpeg stdout, extract JPEG frames
    def reader_thread():
        buffer = bytearray()
        try:
            while not stop_event.is_set():
                rlist, _, _ = select.select([ffmpeg_proc.stdout], [], [], 0.5)
                if not rlist:
                    if stop_event.is_set():
                        break
                    continue
                try:
                    chunk = ffmpeg_proc.stdout.read(65536)
                except Exception:
                    time.sleep(0.01)
                    continue
                if not chunk:
                    time.sleep(0.01)
                    continue
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b'\xff\xd8')
                    if start == -1:
                        break
                    end = buffer.find(b'\xff\xd9', start + 2)
                    if end == -1:
                        break
                    jpeg = bytes(buffer[start:end+2])
                    try:
                        if jpeg_queue.full():
                            try:
                                jpeg_queue.get_nowait()
                            except Exception:
                                pass
                        jpeg_queue.put_nowait(jpeg)
                    except Exception:
                        pass
                    buffer = buffer[end+2:]
        finally:
            stop_event.set()

    # Decoder thread: convert JPEG bytes to BGR
    def decoder_thread():
        try:
            while not stop_event.is_set():
                try:
                    jpeg = jpeg_queue.get(timeout=0.5)
                except Empty:
                    continue
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                try:
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except Exception:
                            pass
                    frame_queue.put_nowait(img)
                except Exception:
                    pass
        finally:
            stop_event.set()

    reader = threading.Thread(target=reader_thread, daemon=True)
    decoder = threading.Thread(target=decoder_thread, daemon=True)
    reader.start()
    decoder.start()

    try:
        while not stop_event.is_set():
            # Get one decoded frame
            try:
                frame = frame_queue.get(timeout=1.0)
            except Empty:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            h, w = frame.shape[:2]

            # minimal overlay (display only)
            cv2.putText(frame, f"ZoomCam (Display Only) - Res: {w}x{h}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            ffmpeg_proc.kill()
        except Exception:
            pass
        cv2.destroyAllWindows()


# =========================
# Main
# =========================
if __name__ == '__main__':
    target_queue = Queue(maxsize=1)
    p1 = Process(target=arducam_process, args=(target_queue,))
    # p2 = Process(target=zoomcam_process, args=(target_queue,))
    p1.start()
    # p2.start()
    p1.join()
    # p2.join()