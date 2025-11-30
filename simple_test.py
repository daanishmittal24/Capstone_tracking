#!/usr/bin/env python3
"""
Simple Face Tracking System - CORRECTED LOGIC + Multithreading
- Threaded camera capture
- Threaded serial sending
"""

import cv2
import numpy as np
import serial
import time
from picamera2 import Picamera2
from threading import Thread, Lock, Event
import queue

# ==================== CONFIG ====================
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_INDEX = 0  # default port (0 = Arducam via rpicam stack)

DEG_PER_PIXEL_X = 0.05
DEG_PER_PIXEL_Y = 0.05

DEAD_ZONE = 15  # pixels


def init_rpicam(camera_index: int = CAMERA_INDEX):
    """Configure Picamera2 similarly to `rpicam-hello --autofocus-mode continuous`."""
    print("Configuring Picamera2 preview (RGB888) with continuous autofocus...")
    try:
        cam = Picamera2(camera_num=camera_index)
    except TypeError:
        # Older Picamera2 builds may not accept camera_num
        cam = Picamera2()

    cfg = cam.create_preview_configuration(
        main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
    )
    cam.configure(cfg)
    cam.start()

    try:
        cam.set_controls({"AfMode": 2, "AfTrigger": 0})
        print("✅ Continuous autofocus requested (AfMode=2, AfTrigger=0)")
    except Exception as exc:
        print(f"⚠️ Autofocus controls not supported on this camera: {exc}")

    return cam

# ==================== THREADED CAMERA ====================
class CameraThread:
    def __init__(self, picam2):
        self.picam2 = picam2
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = Event()
        self.thread = Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            frame = self.picam2.capture_array()
            if frame is None:
                continue
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)

# ==================== THREADED SERIAL SENDER ====================
class SerialSender:
    def __init__(self, port, baud):
        self.ser = None
        self.cmd_queue = queue.Queue(maxsize=5)
        self.lock = Lock()
        self.stop_event = Event()
        self.thread = Thread(target=self._loop, daemon=True)

        try:
            self.ser = serial.Serial(port, baud, timeout=0.5)
            time.sleep(2)
            print("\nArduino messages:")
            time.sleep(0.5)
            while self.ser.in_waiting:
                msg = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if msg:
                    print(f"  {msg}")
            print("\n✅ Serial ready")
        except Exception as e:
            print(f"⚠️ Serial failed: {e}")
            self.ser = None

        if self.ser:
            self.thread.start()

    def send(self, deg_x, deg_y):
        if not self.ser:
            return
        try:
            if self.cmd_queue.full():
                self.cmd_queue.get_nowait()
            self.cmd_queue.put_nowait((deg_x, deg_y))
        except queue.Full:
            pass

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                deg_x, deg_y = self.cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            b = int(round(deg_x))
            w = int(round(deg_y))
            if b == 0 and w == 0:
                continue

            cmd = f"B{b} W{w}\n"
            with self.lock:
                try:
                    self.ser.write(cmd.encode('utf-8'))
                    print(f"→ {cmd.strip()}")
                    time.sleep(0.01)
                    while self.ser.in_waiting:
                        resp = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        if resp:
                            print(f"← {resp}")
                except Exception as e:
                    print(f"⚠️ Serial write failed: {e}")

    def close(self):
        self.stop_event.set()
        if self.ser:
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass
            self.ser.close()

# ==================== KALMAN FILTER ====================
class KalmanFilter2D:
    def __init__(self):
        self.state = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        self.R = np.eye(2, dtype=np.float32) * 10

        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)

        self.initialized = False

    def update(self, cx, cy):
        z = np.array([cx, cy], dtype=np.float32)

        if not self.initialized:
            self.state[0:2] = z
            self.initialized = True
            return float(cx), float(cy)

        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

        return float(self.state[0]), float(self.state[1])

# ==================== FACE TRACKER ====================
class FaceTracker:
    def __init__(self):
        print("="*60)
        print("Starting Simple Face Tracker (Multithreaded)...")
        print("="*60)

        # Camera
        self.picam2 = init_rpicam()
        time.sleep(1)
        print("✅ Camera ready (RGB888 + continuous AF)")

        # Threaded camera
        self.cam_thread = CameraThread(self.picam2)
        self.cam_thread.start()

        # Face detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        print("✅ Detector ready")

        # Kalman filter
        self.kalman = KalmanFilter2D()

        # Threaded serial sender
        self.serial_sender = SerialSender(SERIAL_PORT, BAUD_RATE)

        self.center_x = CAMERA_WIDTH // 2
        self.center_y = CAMERA_HEIGHT // 2

        print("="*60)
        print("SYSTEM READY")
        print("="*60)

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50)
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx = x + w // 2
            cy = y + h // 2
            return cx, cy, (x, y, w, h)
        return None

    def run(self):
        print("\n" + "="*60)
        print("FACE TRACKING ACTIVE")
        print("="*60)
        print("Logic:")
        print("  Face LEFT  → Camera LEFT  (B-)")
        print("  Face RIGHT → Camera RIGHT (B+)")
        print("  Face UP    → Camera UP    (W-)")
        print("  Face DOWN  → Camera DOWN  (W+)")
        print()
        print("Press 'q' to quit")
        print("="*60 + "\n")

        try:
            while True:
                frame = self.cam_thread.get_frame()
                if frame is None:
                    continue

                display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                detection = self.detect_face(frame)

                if detection is None:
                    cv2.putText(display, "No face", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cx, cy, (x, y, w, h) = detection

                    # Kalman filter
                    cx_f, cy_f = self.kalman.update(cx, cy)
                    fx, fy = int(cx_f), int(cy_f)

                    # Draw detection and center
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(display, (fx, fy), 5, (0, 255, 255), -1)
                    cv2.putText(display, "Kalman", (fx + 10, fy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                    cv2.line(display, (self.center_x - 20, self.center_y),
                             (self.center_x + 20, self.center_y), (255, 0, 0), 2)
                    cv2.line(display, (self.center_x, self.center_y - 20),
                             (self.center_x, self.center_y + 20), (255, 0, 0), 2)

                    # Signed pixel error (face relative to optical center)
                    dx = fx - self.center_x  # +dx → face on right side
                    dy = fy - self.center_y  # +dy → face below center

                    cv2.arrowedLine(display, (self.center_x, self.center_y), (fx, fy),
                                    (255, 255, 0), 2, tipLength=0.3)

                    if abs(dx) < DEAD_ZONE and abs(dy) < DEAD_ZONE:
                        cv2.putText(display, "CENTERED", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Arduino sign convention (from VisionLock sketch):
                        # BASE:  +deg = anticlockwise (turns RIGHT)
                        #        -deg = clockwise    (turns LEFT)
                        # WHEEL: +deg = clockwise    (tilt DOWN)
                        #        -deg = anticlockwise (tilt UP)
                        # Therefore send proportional degrees using dx, dy directly.
                        # NOTE: Reversed base sign to match actual hardware behavior
                        deg_x = -dx * DEG_PER_PIXEL_X
                        deg_y = dy * DEG_PER_PIXEL_Y

                        self.serial_sender.send(deg_x, deg_y)

                        camera_move = ""
                        if dx > 0:
                            camera_move += "Cam RIGHT "
                        elif dx < 0:
                            camera_move += "Cam LEFT "
                        if dy > 0:
                            camera_move += "Cam DOWN"
                        elif dy < 0:
                            camera_move += "Cam UP"

                        cv2.putText(display, f"B={deg_x:.1f} W={deg_y:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(display, camera_move, (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        cv2.putText(display, f"dx={dx:.0f} dy={dy:.0f}", (10, 85),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.imshow("Face Tracker", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            print("\nShutting down...")
            self.cam_thread.stop()
            self.picam2.close()
            self.serial_sender.close()
            cv2.destroyAllWindows()
            print("✅ Shutdown complete")

# ==================== MAIN ====================
if __name__ == "__main__":
    try:
        tracker = FaceTracker()
        tracker.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
