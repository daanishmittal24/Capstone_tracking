
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
# Helpers - Using OpenCV face detection instead of MediaPipe
# =========================
def detect_face_center(frame):
    """Detect face using OpenCV Haar Cascade and return center coordinates"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    if len(faces) > 0:
        # Use the largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        cx = x + w // 2
        cy = y + h // 2
        return cx, cy, x, y, x + w, y + h
    
    return None

def detect_upper_body_center(frame):
    """Detect upper body using OpenCV Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load upper body detector
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    
    bodies = upper_body_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(50, 50)
    )
    
    if len(bodies) > 0:
        # Use the largest detection
        bodies = sorted(bodies, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = bodies[0]
        cx = x + w // 2
        cy = y + h // 2
        return cx, cy, x, y, x + w, y + h
    
    return None

def compute_box_center(frame, mode):
    """Return (cx, cy, x_min, y_min, x_max, y_max) using OpenCV detection"""
    if mode == "face":
        return detect_face_center(frame)
    elif mode == "upper_body":
        return detect_upper_body_center(frame)
    elif mode == "upper_body_face":
        # Try face first, then upper body
        result = detect_face_center(frame)
        if result is None:
            result = detect_upper_body_center(frame)
        return result
    return None

def arducam_process(target_queue: Queue):
    """
    Simplified Arducam: Picamera2 with native continuous autofocus (no external rpicam calls).
    Uses built-in AfMode=Continuous for seamless AF during capture.
    """
    import time, cv2, numpy as np, os
    from libcamera import controls  # Required for AfModeEnum

    cam_index = 1
    window_name = "Arducam View"
    mode = "upper_body_face"

    kf = KalmanTracker()
    PROC_W, PROC_H = 640, 360  # detection/downscale size
    DISP_W, DISP_H = 640, 360  # preview size

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
        # High-res main stream + continuous AF (no other controls needed)
        config = picam2.create_video_configuration(
            main={"format": "RGB888", "size": (4096, 3074)},
            controls={"AfMode": controls.AfModeEnum.Continuous}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1.0)  # Let AF initialize
        print("✅ Arducam: Continuous AF enabled via Picamera2.")
    except Exception as e:
        print("❌ Camera init failed:", e)
        return

    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                continue

            h, w, _ = frame.shape

            # Downscale for detection (fast)
            small = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
            info = compute_box_center(small, mode)

            if info is not None:
                cx_s, cy_s, x_min_s, y_min_s, x_max_s, y_max_s = info

                # Map to original coords
                sx = float(w) / float(PROC_W)
                sy = float(h) / float(PROC_H)
                cx = cx_s * sx
                cy = cy_s * sy
                x1 = int(x_min_s * sx)
                x2 = int(x_max_s * sx)
                y1 = int(y_min_s * sy)
                y2 = int(y_max_s * sy)

                # Kalman smoothing
                cx_f, cy_f = kf.update(cx, cy)

                # Draw on full-res frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx_f), int(cy_f)), 4, (0, 0, 255), -1)

                # Send normalized target to queue
                nx = np.clip(cx_f / w, 0.0, 1.0)
                ny = np.clip(cy_f / h, 0.0, 1.0)
                try:
                    while not target_queue.empty():
                        target_queue.get_nowait()
                except Exception:
                    pass
                target_queue.put((nx, ny))

            # Lightweight preview
            disp = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)
            cv2.putText(disp, "AF: Picamera2 Continuous", (8, 20),
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
        cv2.destroyWindow(window_name)



# =========================
# ZoomCam process
# =========================
# Replace your current zoomcam_process with this optimized threaded version.
def zoomcam_process(target_queue: Queue):
    import subprocess, cv2, time, numpy as np, serial, os
    import threading, select
    from queue import Queue as ThreadQueue, Empty

    window_name = "ZoomCam View"
    mode = "face"

    # Adjust these for tradeoff between quality and CPU
    OUT_WIDTH, OUT_HEIGHT = 640, 360     # ffmpeg output resolution (reduce to lower load)
    OUT_FPS = 15                         # lower fps reduces CPU
    JPEG_QUEUE_MAX = 2
    PROCESS_SKIP = 0                     # process every frame; set >0 to skip frames (e.g. 1 -> process every other)
    DRAW_EVERY = 1                       # draw every N processed frames (reduce GUI cost)

    # FFmpeg: request a smaller MJPEG stream (less CPU + bandwidth)
    ffmpeg_cmd = [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-i", "udp://@:5000",
        "-an",
        "-c:v", "mjpeg",
        "-q:v", "8",                            # increase number for lower quality / less CPU/bandwidth (8 is low->med)
        "-vf", f"scale={OUT_WIDTH}:{OUT_HEIGHT}",  # force resolution
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

    # optional serial
    try:
        ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        time.sleep(1)
        print("✅ [ZoomCam] Serial opened")
    except Exception as e:
        print(f"⚠️ [ZoomCam] Serial unavailable: {e}")
        ser = None

    # Preload cascades once (avoid recreating each frame)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

    # threaded queues
    jpeg_queue = ThreadQueue(maxsize=JPEG_QUEUE_MAX)   # holds raw JPEG bytes
    frame_queue = ThreadQueue(maxsize=JPEG_QUEUE_MAX)  # holds decoded BGR frames for processing

    stop_event = threading.Event()

    # Reader thread: read ffmpeg stdout, extract JPEG frames and push raw JPEG bytes into jpeg_queue
    def reader_thread():
        buffer = bytearray()
        try:
            while not stop_event.is_set():
                # use select to wait for data with small timeout
                rlist, _, _ = select.select([ffmpeg_proc.stdout], [], [], 0.5)
                if not rlist:
                    # no data, loop
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
                # find SOI/EOI
                while True:
                    start = buffer.find(b'\xff\xd8')
                    if start == -1:
                        # no start yet
                        break
                    end = buffer.find(b'\xff\xd9', start + 2)
                    if end == -1:
                        # wait for more
                        break
                    jpeg = bytes(buffer[start:end+2])
                    # drop old frames if queue is full to keep latency low
                    try:
                        if jpeg_queue.full():
                            try:
                                jpeg_queue.get_nowait()
                            except Exception:
                                pass
                        jpeg_queue.put_nowait(jpeg)
                    except Exception:
                        pass
                    # remove used bytes
                    buffer = buffer[end+2:]
        finally:
            stop_event.set()

    # Decoder/producer: convert JPEG bytes to BGR and push to frame_queue
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
                # push decoded frame (drop if processing queue full)
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

    # Tracking + control (main worker loop)
    kf = KalmanTracker()
    BASE_SIGN = 1
    WHEEL_SIGN = 1
    DEAD_BAND_NORM = 0.02
    KP_BASE = 30
    KP_WHEEL = 16
    MAX_STEP_BASE = 10
    MAX_STEP_WHEEL = 5
    SEND_PERIOD = 0.06
    EMA_ALPHA = 0.25
    err_x_ema, err_y_ema = 0.0, 0.0
    last_send = 0.0
    proc_counter = 0
    draw_counter = 0

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
            ser.write(cmd.encode('utf-8'))
        except Exception as e:
            print(f"⚠️ [ZoomCam] Serial write failed: {e}")
        last_send = now

    try:
        while not stop_event.is_set():
            # Get one decoded frame (blocks a bit)
            try:
                frame = frame_queue.get(timeout=1.0)
            except Empty:
                # no frames yet
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            proc_counter += 1
            # optional frame skip
            if PROCESS_SKIP > 0 and (proc_counter % (PROCESS_SKIP + 1)) != 0:
                continue

            h, w = frame.shape[:2]

            # small detection image to reduce compute (same strategy as your Arducam)
            PROC_W, PROC_H = 320, 180
            small = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

            # detection: try face then upper body
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            bodies = []
            if len(faces) == 0:
                bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

            if len(faces) > 0:
                x, y, ww, hh = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
            elif len(bodies) > 0:
                x, y, ww, hh = sorted(bodies, key=lambda r: r[2]*r[3], reverse=True)[0]
            else:
                x = y = ww = hh = None

            have_target = False
            if x is not None:
                # map back to full resolution
                scale_x = w / PROC_W
                scale_y = h / PROC_H
                cx = (x + ww//2) * scale_x
                cy = (y + hh//2) * scale_y
                cx_f, cy_f = kf.update(cx, cy)
                have_target = True
                # draw only occasionally to reduce cost
                draw_counter += 1
                if draw_counter % DRAW_EVERY == 0:
                    x_min = int(x*scale_x)
                    x_max = int((x+ww)*scale_x)
                    y_min = int(y*scale_y)
                    y_max = int((y+hh)*scale_y)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
                    cv2.circle(frame, (int(cx_f), int(cy_f)), 4, (0,0,255), -1)
            else:
                # fallback: use hint from Arducam if available
                try:
                    nx, ny = target_queue.get_nowait()
                    cx_hint = int(nx * w)
                    cy_hint = int(ny * h)
                    cx_f, cy_f = kf.update(cx_hint, cy_hint)
                    have_target = True
                    if draw_counter % DRAW_EVERY == 0:
                        cv2.drawMarker(frame, (int(cx_f), int(cy_f)), (0,255,255),
                                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                except Exception:
                    have_target = False

            if have_target:
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

            # minimal overlay
            cv2.putText(frame, f"Res: {w}x{h}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

            # show GUI but reduce wait time
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
        if ser:
            try:
                ser.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


# =========================
# Main
# =========================
if __name__ == '__main__':
    target_queue = Queue(maxsize=1)
    p1 = Process(target=arducam_process, args=(target_queue,))
    p2 = Process(target=zoomcam_process, args=(target_queue,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()