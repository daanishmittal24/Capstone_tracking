#!/usr/bin/env python3
"""
track_dual_independent.py

DUAL-CAMERA COORDINATED FACE TRACKING
Compares face positions in both cameras to determine vertical movement

Architecture:
  - Process 1 (Arducam): Detects face → Controls BASE (horizontal) + shares face Y position
  - Process 2 (ZoomCam): Detects face → Compares with Arducam face Y → Controls WHEEL (vertical)
  
Physical Setup:
  Arducam (1280x720) → BASE motor (horizontal rotation) + face position sharing
  ZoomCam (640x360)  → WHEEL motor (vertical tilt)
  
Strategy:
  - Arducam has wider FOV, detects face and shares vertical position
  - ZoomCam also detects face in its narrow view
  - Compare face Y positions: if Arducam face is higher/lower than ZoomCam, adjust WHEEL
  - Laser shown in Arducam as visual indicator only (not used for control)

Performance:
  - Multi-process architecture (no GIL)
  - Multi-threaded FFmpeg receiver
  - Shared face position via Manager dict
  - Coordinated vertical tracking based on face comparison

Controls (either window):
  q = quit both processes
"""
import os
import time
from pathlib import Path
from multiprocessing import Process, Event as MPEvent, Manager
import subprocess
import threading
import select
from queue import Queue as ThreadQueue, Empty

import numpy as np
import cv2
import serial
from picamera2 import Picamera2

# ========== CONFIG ==========
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

# Arducam (horizontal tracking)
ARDUCAM_WIDTH = 1280
ARDUCAM_HEIGHT = 720
ARDUCAM_DISPLAY_WIDTH = 640
ARDUCAM_DISPLAY_HEIGHT = 360

# ZoomCam (vertical tracking)
ZOOMCAM_WIDTH = 640
ZOOMCAM_HEIGHT = 360
UDP_PORT = 5000
OUT_FPS = 20

# Face Detection
FACE_MODEL_PATH = 'models/face_detection_yunet_2023mar.onnx'

# Control Parameters
BASE_SIGN = -1  # Use direct proportional control (reversed in calculation)
WHEEL_SIGN = -1  # REVERSED - correct vertical direction
DEAD_ZONE_PIXELS_X = 50  # Horizontal dead zone (pixels) - don't move if within this
DEAD_ZONE_PIXELS_Y = 60  # Vertical dead zone - INCREASED to reduce movement (was 30)
KP_BASE = 0.03  # Reduced gain for slower movement (was 0.05)
KP_WHEEL_VERTICAL = 0.005  # VERY SMALL gain for vertical - tiny steps (was 0.01)
MAX_STEP_BASE = 5   # Reduced max step for smoother movement (was 10)
MAX_STEP_WHEEL = 2  # VERY SMALL max step for vertical - 2 degrees max (was 3)
SEND_PERIOD = 0.25  # Slower commands for gradual movement (was 0.2)
SETTLING_DELAY = 0.35  # Longer settling for small steps (was 0.3)

# Laser Detection (visual indicator only - not used for control)
LASER_MIN_BRIGHTNESS = 200  # Minimum brightness for laser detection
LASER_MIN_AREA = 5  # Minimum area in pixels
LASER_MAX_AREA = 500  # Maximum area in pixels
# ============================


# ========== SHARED SERIAL CONNECTION ==========
# Global serial object and lock (shared across processes via Manager)
serial_conn = None
serial_lock = None
shared_face_state = None  # Shared dict for face position comparison

def init_serial():
    """Initialize serial connection once at startup"""
    global serial_conn
    try:
        serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        while serial_conn.in_waiting:
            serial_conn.readline()
        print('[Serial] ✅ Port opened')
        return True
    except Exception as e:
        print(f'[Serial] ❌ Failed: {e}')
        return False


def detect_laser(frame_bgr):
    """
    Detect bright laser point in frame (VISUAL INDICATOR ONLY)
    Returns (cx, cy) of laser centroid or None
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Threshold for bright spots
    _, bright = cv2.threshold(gray, LASER_MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest bright spot within area limits
    valid_contours = [c for c in contours if LASER_MIN_AREA <= cv2.contourArea(c) <= LASER_MAX_AREA]
    if not valid_contours:
        return None
    
    largest = max(valid_contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    
    if M['m00'] == 0:
        return None
    
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    return (cx, cy)


# ========== ARDUCAM PROCESS (Horizontal Tracking + Face Position Sharing) ==========
def arducam_process(serial_lock, stop_event, shared_face_state):
    """
    Arducam Process - Horizontal (BASE) tracking + Face position sharing
    Detects face on Arducam → controls BASE + shares face Y position for vertical coordination
    Also shows laser as visual indicator (not used for control)
    """
    print('[Arducam] Starting horizontal tracking + face position sharing...')
    
    # === Face Detector ===
    if not Path(FACE_MODEL_PATH).exists():
        print(f'[Arducam] ❌ Face model not found: {FACE_MODEL_PATH}')
        stop_event.set()
        return
    
    try:
        face_detector = cv2.FaceDetectorYN.create(
            FACE_MODEL_PATH, "", (ARDUCAM_WIDTH, ARDUCAM_HEIGHT),
            score_threshold=0.5, nms_threshold=0.3
        )
        print('[Arducam] ✅ Face detector loaded')
    except Exception as e:
        print(f'[Arducam] ❌ Face detector failed: {e}')
        stop_event.set()
        return
    
    # === Camera ===
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (ARDUCAM_WIDTH, ARDUCAM_HEIGHT)}
        )
        picam2.configure(config)
        picam2.start()
        picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
        time.sleep(1)
        print('[Arducam] ✅ Camera started')
    except Exception as e:
        print(f'[Arducam] ❌ Camera failed: {e}')
        stop_event.set()
        return
    
    # === Window ===
    window = 'Arducam - Horizontal Tracking (BASE)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    
    # === State ===
    frame_count = 0
    fps_start = time.time()
    current_fps = 0
    last_send_time = 0.0
    
    def send_base_cmd(delta_deg):
        """Send BASE command directly to serial (with lock to prevent conflicts)"""
        nonlocal last_send_time
        delta = int(np.clip(delta_deg, -MAX_STEP_BASE, MAX_STEP_BASE))
        if delta == 0:
            return
        
        # Rate limiting - don't send too frequently
        now = time.time()
        if now - last_send_time < SEND_PERIOD:
            return
        
        try:
            with serial_lock:
                cmd = f"B{delta} W0"
                serial_conn.write((cmd + '\n').encode('utf-8'))
                print(f'[Arducam] → {cmd}')
                last_send_time = now
                # Add settling delay to prevent immediate reversal
                time.sleep(SETTLING_DELAY)
        except Exception as e:
            print(f'[Arducam] Serial error: {e}')
    
    print('[Arducam] 🎯 Tracking horizontal (X-axis) + sharing face position for vertical coordination')
    print('[Arducam] Press "q" to quit')
    
    # === Main Loop ===
    try:
        while not stop_event.is_set():
            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                continue
            
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]
            
            # FPS
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                current_fps = frame_count
                frame_count = 0
                fps_start = time.time()
            
            # === LASER DETECTION (VISUAL INDICATOR ONLY - not used for control) ===
            laser_pos = detect_laser(frame_bgr)
            
            # === FACE DETECTION (for horizontal tracking + position sharing) ===
            _, faces = face_detector.detect(frame_bgr)
            
            have_face = False
            if faces is not None and len(faces) > 0:
                # Get largest face
                best_idx = int(np.argmax(faces[:, 4]))
                x, y, box_w, box_h, score = faces[best_idx, :5]
                
                x, y = int(x), int(y)
                box_w, box_h = int(box_w), int(box_h)
                cx = x + box_w // 2
                cy = y + box_h // 2
                have_face = True
                
                # SHARE FACE POSITION with ZoomCam for vertical coordination
                # Calculate normalized Y position (0.0 = top, 0.5 = center, 1.0 = bottom)
                face_y_normalized = cy / h
                shared_face_state['arducam_face_y'] = face_y_normalized
                shared_face_state['arducam_cy'] = cy
                shared_face_state['arducam_height'] = h
                shared_face_state['timestamp'] = time.time()
                shared_face_state['has_face'] = True
                have_face = True
                
                # Calculate horizontal offset (like calibration script)
                offset_x = cx - w // 2
                
                # Only move if outside dead zone
                if abs(offset_x) > DEAD_ZONE_PIXELS_X:
                    # Use reduced gain for slower, smoother movement
                    move_base = int(offset_x * -KP_BASE)  # Now using KP_BASE=0.03
                    # Clamp to MAX_STEP_BASE (5 degrees for smoother movement)
                    move_base = max(-MAX_STEP_BASE, min(MAX_STEP_BASE, move_base))
                    
                    if abs(move_base) >= 1:
                        send_base_cmd(move_base)
                
                # Draw on frame
                disp = cv2.resize(frame_bgr, (ARDUCAM_DISPLAY_WIDTH, ARDUCAM_DISPLAY_HEIGHT))
                scale_x = ARDUCAM_DISPLAY_WIDTH / w
                scale_y = ARDUCAM_DISPLAY_HEIGHT / h
                
                dx = int(x * scale_x)
                dy = int(y * scale_y)
                dw = int(box_w * scale_x)
                dh = int(box_h * scale_y)
                dcx = int(cx * scale_x)
                dcy = int(cy * scale_y)
                
                cv2.rectangle(disp, (dx, dy), (dx+dw, dy+dh), (0, 255, 0), 2)
                cv2.circle(disp, (dcx, dcy), 5, (0, 0, 255), -1)
                
                # Show horizontal tracking line
                cv2.line(disp, (ARDUCAM_DISPLAY_WIDTH//2, dcy), (dcx, dcy), (0, 255, 255), 2)
                cv2.putText(disp, f'Face X: {offset_x:+d}px', (dcx+10, dcy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                # Show face Y position
                cv2.putText(disp, f'Face Y: {face_y_normalized:.2f} (sharing)', (dcx+10, dcy+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
            else:
                disp = cv2.resize(frame_bgr, (ARDUCAM_DISPLAY_WIDTH, ARDUCAM_DISPLAY_HEIGHT))
                scale_x = ARDUCAM_DISPLAY_WIDTH / w
                scale_y = ARDUCAM_DISPLAY_HEIGHT / h
                # Clear face state when no face detected
                shared_face_state['has_face'] = False
            
            # Draw laser position if detected (VISUAL INDICATOR ONLY)
            if laser_pos:
                dlaser_cx = int(laser_pos[0] * scale_x)
                dlaser_cy = int(laser_pos[1] * scale_y)
                cv2.circle(disp, (dlaser_cx, dlaser_cy), 8, (128, 128, 128), 2)
                cv2.circle(disp, (dlaser_cx, dlaser_cy), 3, (255, 255, 255), -1)
                cv2.putText(disp, 'Laser (visual only)', (dlaser_cx+10, dlaser_cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # UI overlay
            cv2.putText(disp, 'ARDUCAM - Horizontal + Face Sharing', (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(disp, f'{w}x{h} @ {current_fps} FPS', (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            face_status = f'Face Y: {face_y_normalized:.2f}' if have_face else 'No face'
            cv2.putText(disp, f'Sharing: {face_status}', (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Center lines
            cv2.line(disp, (ARDUCAM_DISPLAY_WIDTH//2, 0), 
                    (ARDUCAM_DISPLAY_WIDTH//2, ARDUCAM_DISPLAY_HEIGHT), (255, 0, 0), 1)
            cv2.line(disp, (0, ARDUCAM_DISPLAY_HEIGHT//2), 
                    (ARDUCAM_DISPLAY_WIDTH, ARDUCAM_DISPLAY_HEIGHT//2), (255, 0, 0), 1)
            
            cv2.imshow(window, disp)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('[Arducam] Quit')
                stop_event.set()
                break
    
    finally:
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyWindow(window)
        print('[Arducam] Stopped')


# ========== ZOOMCAM PROCESS (Vertical Tracking) ==========
def zoomcam_process(serial_lock, stop_event, shared_face_state):
    """
    ZoomCam Process - Vertical (WHEEL) tracking by comparing face positions
    Compares face Y position in ZoomCam vs Arducam to determine up/down movement
    """
    print('[ZoomCam] Starting face-comparison vertical tracking...')
    
    # === Face Detector ===
    if not Path(FACE_MODEL_PATH).exists():
        print(f'[ZoomCam] ❌ Face model not found: {FACE_MODEL_PATH}')
        stop_event.set()
        return
    
    try:
        face_detector = cv2.FaceDetectorYN.create(
            FACE_MODEL_PATH, "", (ZOOMCAM_WIDTH, ZOOMCAM_HEIGHT),
            score_threshold=0.5, nms_threshold=0.3
        )
        print('[ZoomCam] ✅ Face detector loaded')
    except Exception as e:
        print(f'[ZoomCam] ❌ Face detector failed: {e}')
        stop_event.set()
        return
    
    # === FFmpeg ===
    ffmpeg_cmd = [
        "ffmpeg", "-fflags", "nobuffer", "-flags", "low_delay",
        "-i", f"udp://@:{UDP_PORT}", "-an", "-c:v", "mjpeg",
        "-q:v", "5", "-vf", f"scale={ZOOMCAM_WIDTH}:{ZOOMCAM_HEIGHT}",
        "-r", str(OUT_FPS), "-f", "image2pipe", "-"
    ]
    
    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, bufsize=10**6)
        if proc.stdout is None:
            raise RuntimeError("FFmpeg stdout unavailable")
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except:
            pass
        print('[ZoomCam] ✅ FFmpeg started')
    except Exception as e:
        print(f'[ZoomCam] ❌ FFmpeg failed: {e}')
        stop_event.set()
        return
    
    # === Multi-threaded receiver ===
    jpeg_queue = ThreadQueue(maxsize=2)
    frame_queue = ThreadQueue(maxsize=2)
    stream_stop = threading.Event()
    
    def reader_thread():
        buf = bytearray()
        try:
            while not stream_stop.is_set() and not stop_event.is_set():
                r, _, _ = select.select([proc.stdout], [], [], 0.5)
                if not r:
                    continue
                try:
                    chunk = proc.stdout.read(65536)
                except:
                    time.sleep(0.01)
                    continue
                if not chunk:
                    time.sleep(0.01)
                    continue
                buf.extend(chunk)
                while True:
                    s = buf.find(b'\xff\xd8')
                    if s == -1:
                        break
                    e = buf.find(b'\xff\xd9', s+2)
                    if e == -1:
                        break
                    jpeg = bytes(buf[s:e+2])
                    try:
                        if jpeg_queue.full():
                            try:
                                jpeg_queue.get_nowait()
                            except:
                                pass
                        jpeg_queue.put_nowait(jpeg)
                    except:
                        pass
                    buf = buf[e+2:]
        finally:
            stream_stop.set()
    
    def decoder_thread():
        try:
            while not stream_stop.is_set() and not stop_event.is_set():
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
                        except:
                            pass
                    frame_queue.put_nowait(img)
                except:
                    pass
        finally:
            stream_stop.set()
    
    reader = threading.Thread(target=reader_thread, daemon=True)
    decoder = threading.Thread(target=decoder_thread, daemon=True)
    reader.start()
    decoder.start()
    
    # === Window ===
    window = 'ZoomCam - Vertical Tracking (WHEEL)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    
    # === State ===
    frame_count = 0
    fps_start = time.time()
    current_fps = 0
    last_send_time = 0.0
    
    def send_wheel_cmd(delta_deg):
        """Send WHEEL command directly to serial (with lock to prevent conflicts)"""
        nonlocal last_send_time
        delta = int(np.clip(delta_deg, -MAX_STEP_WHEEL, MAX_STEP_WHEEL))
        if delta == 0:
            return
        
        # Rate limiting - don't send too frequently
        now = time.time()
        if now - last_send_time < SEND_PERIOD:
            return
        
        try:
            with serial_lock:
                cmd = f"B0 W{delta}"
                serial_conn.write((cmd + '\n').encode('utf-8'))
                print(f'[ZoomCam] → {cmd}')
                last_send_time = now
                # Add settling delay to prevent immediate reversal
                time.sleep(SETTLING_DELAY)
        except Exception as e:
            print(f'[ZoomCam] Serial error: {e}')
    
    print('[ZoomCam] 🎯 Comparing face positions for vertical tracking')
    print('[ZoomCam] Press "q" to quit')
    
    # === Main Loop ===
    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1.0)
            except Empty:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
                continue
            
            h, w = frame.shape[:2]
            
            # FPS
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                current_fps = frame_count
                frame_count = 0
                fps_start = time.time()
            
            # === GET ARDUCAM FACE POSITION ===
            arducam_has_face = shared_face_state.get('has_face', False)
            arducam_face_y_norm = shared_face_state.get('arducam_face_y', 0.5)
            arducam_timestamp = shared_face_state.get('timestamp', 0)
            
            # Check if Arducam data is fresh (within last 0.5 seconds)
            data_is_fresh = (time.time() - arducam_timestamp) < 0.5
            
            # === DETECT FACE IN ZOOMCAM ===
            _, faces = face_detector.detect(frame)
            
            zoomcam_has_face = False
            zoomcam_face_y_norm = 0.5
            
            if faces is not None and len(faces) > 0:
                # Get largest face
                best_idx = int(np.argmax(faces[:, 4]))
                x, y, box_w, box_h, score = faces[best_idx, :5]
                
                x, y = int(x), int(y)
                box_w, box_h = int(box_w), int(box_h)
                cx = x + box_w // 2
                cy = y + box_h // 2
                zoomcam_has_face = True
                
                # Calculate normalized Y position for ZoomCam face
                zoomcam_face_y_norm = cy / h
                
                # Draw face
                cv2.rectangle(frame, (x, y), (x+box_w, y+box_h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'ZoomCam Y: {zoomcam_face_y_norm:.2f}', (cx+10, cy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # === COMPARE FACE POSITIONS AND CONTROL WHEEL ===
            if arducam_has_face and zoomcam_has_face and data_is_fresh:
                # Compare normalized Y positions
                # If Arducam face is HIGHER (smaller Y) than ZoomCam face → move UP (negative)
                # If Arducam face is LOWER (larger Y) than ZoomCam face → move DOWN (positive)
                y_diff = arducam_face_y_norm - zoomcam_face_y_norm
                
                # Convert to pixel difference for dead zone check
                y_diff_pixels = int(y_diff * h)
                
                # Only move if difference is significant
                if abs(y_diff_pixels) > DEAD_ZONE_PIXELS_Y:
                    # Move to align: if Arducam is higher, move camera up (negative WHEEL)
                    # VERY SMALL sensitivity multiplier (15) for tiny, gradual steps
                    move_wheel = int(y_diff_pixels * -KP_WHEEL_VERTICAL * 15)
                    move_wheel = max(-MAX_STEP_WHEEL, min(MAX_STEP_WHEEL, move_wheel))
                    
                    if abs(move_wheel) >= 1:
                        send_wheel_cmd(move_wheel)
                
                # Show comparison visualization
                arducam_y_in_frame = int(arducam_face_y_norm * h)
                cv2.line(frame, (0, arducam_y_in_frame), (w, arducam_y_in_frame), (255, 128, 0), 2)
                cv2.putText(frame, f'Arducam Y: {arducam_face_y_norm:.2f}', (10, arducam_y_in_frame-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
                cv2.putText(frame, f'Diff: {y_diff:+.2f} ({y_diff_pixels:+d}px)', (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # UI overlay
            cv2.putText(frame, 'ZOOMCAM - Face Comparison Tracking', (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f'{w}x{h} @ {current_fps} FPS', (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Status based on face detection in both cameras
            if arducam_has_face and zoomcam_has_face and data_is_fresh:
                status = f'Comparing faces'
            elif zoomcam_has_face:
                status = 'ZoomCam face only'
            elif arducam_has_face and data_is_fresh:
                status = 'Arducam face only'
            else:
                status = 'No faces detected'
            
            cv2.putText(frame, f'Mode: {status}', (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Center horizontal line
            cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 1)
            
            cv2.imshow(window, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('[ZoomCam] Quit')
                stop_event.set()
                break
    
    finally:
        stream_stop.set()
        try:
            proc.kill()
        except:
            pass
        cv2.destroyWindow(window)
        print('[ZoomCam] Stopped')


# ========== MAIN ==========
def main():
    global serial_conn, serial_lock, shared_laser_state
    
    print('=' * 80)
    print('DUAL-CAMERA COORDINATED TRACKING')
    print('=' * 80)
    print()
    print('🎯 Architecture:')
    print('  Process 1: Arducam → Detects laser + face → Controls BASE (horizontal)')
    print('                     → Shares laser vertical position')
    print('  Process 2: ZoomCam → Uses Arducam laser guidance → Controls WHEEL (vertical)')
    print()
    print('📺 Two coordinated windows:')
    print('  - Arducam: Wide FOV - detects laser, controls horizontal')
    print('  - ZoomCam: Narrow FOV - uses laser guidance for vertical')
    print()
    print('✅ Coordinated tracking using shared laser position')
    print('Press "q" in either window to quit all processes.')
    print('=' * 80)
    print()
    
    # Initialize serial connection ONCE
    if not init_serial():
        print('❌ Failed to initialize serial. Exiting.')
        return
    
    # Create shared lock and state using Manager for cross-process synchronization
    manager = Manager()
    serial_lock = manager.Lock()
    shared_laser_state = manager.dict()
    shared_laser_state['offset_y'] = None
    shared_laser_state['timestamp'] = 0
    shared_laser_state['frame_height'] = ARDUCAM_HEIGHT
    
    stop_event = MPEvent()
    
    # Start both camera processes with shared state
    p_arducam = Process(target=arducam_process, args=(serial_lock, stop_event, shared_laser_state), name='Arducam')
    p_zoomcam = Process(target=zoomcam_process, args=(serial_lock, stop_event, shared_laser_state), name='ZoomCam')
    
    p_arducam.start()
    p_zoomcam.start()
    
    print('✅ Both processes started with coordinated tracking')
    print()
    
    try:
        p_arducam.join()
        p_zoomcam.join()
    except KeyboardInterrupt:
        print('\n⚠️  Keyboard interrupt')
        stop_event.set()
        p_arducam.terminate()
        p_zoomcam.terminate()
        p_arducam.join(timeout=2)
        p_zoomcam.join(timeout=2)
    finally:
        # Close serial connection
        if serial_conn:
            try:
                serial_conn.close()
                print('[Serial] Connection closed')
            except:
                pass
    
    print('\n✅ All processes stopped')

if __name__ == '__main__':
    main()
