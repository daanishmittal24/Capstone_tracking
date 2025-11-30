#!/usr/bin/env python3
"""
calibrate_vertical_manual.py

MANUAL VERTICAL CALIBRATION - ISOLATED PROCESSES
Low-latency, manual control for precise mapping

How it works:
1. You see both cameras in real-time (ISOLATED PROCESSES)
2. Use arrow keys to move WHEEL up/down in small steps
3. Press ENTER when ZoomCam is pointing where you want
4. System records: WHEEL_degrees → Arducam_Y_position
5. Repeat for multiple positions
6. Press 's' to save mapping

Controls:
  ↑/↓     = Move WHEEL up/down (1° steps)
  w/x     = Move WHEEL up/down (5° steps)
  ENTER   = Record current position
  r       = Return to home
  's'     = Save and quit
  'q'     = Quit without saving
"""
import os
import time
import json
import threading
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
from queue import Queue as ThreadQueue, Empty
import select

import numpy as np
import cv2
import serial
from picamera2 import Picamera2
import subprocess

# ========== CONFIG ==========
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

# Arducam - optimized for speed
ARDUCAM_WIDTH = 1280
ARDUCAM_HEIGHT = 720
ARDUCAM_DISPLAY_WIDTH = 640
ARDUCAM_DISPLAY_HEIGHT = 360

# ZoomCam - optimized
ZOOMCAM_WIDTH = 640
ZOOMCAM_HEIGHT = 360
UDP_PORT = 5000

FACE_MODEL_PATH = 'models/face_detection_yunet_2023mar.onnx'
MAPPING_FILE = 'vertical_mapping.json'
HOMING_SETTLE_DELAY = 2.0  # Extra hold after zeroing to avoid queued commands
MOTION_COMPLETE_TIMEOUT = 20  # Max time to wait for controller to finish a move
HOMING_STATUS_DELAY = 0.5  # Wait before retrying status during homing
HOMING_POST_MOVE_DELAY = 0.8  # Allow extra time after each homing move
WHEEL_STATUS_TOLERANCE = 0.75  # Degrees of mismatch before trusting cached value
# ============================


class FastSerial:
    """Optimized serial communication"""
    def __init__(self, port, baudrate):
        self.conn = serial.Serial(port, baudrate, timeout=0.5)
        self.lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.estimated_wheel_deg = 0.0
        time.sleep(1.5)
        # Clear buffer
        while self.conn.in_waiting:
            self.conn.readline()
        print('[Serial] ✅ Ready')
    
    def _parse_status_line(self, line):
        if 'BASE_DEG' not in line and 'WHEEL_DEG' not in line:
            return None
        result = {}
        try:
            for part in line.split('|'):
                part = part.strip()
                if 'WHEEL_DEG' in part:
                    result['wheel_deg'] = float(part.split('=')[1].strip())
                elif 'BASE_DEG' in part:
                    result['base_deg'] = float(part.split('=')[1].strip())
                elif 'HALL_WHEEL' in part:
                    result['hall_wheel'] = int(part.split('=')[1].strip())
                elif 'HALL_BASE' in part:
                    result['hall_base'] = int(part.split('=')[1].strip())
        except Exception as exc:
            print(f'[Serial] Parse error: {exc}. Raw: {line}')
            return None
        return result if result else None
    
    def _wait_for_motion_complete_locked(self, timeout=MOTION_COMPLETE_TIMEOUT):
        deadline = time.time() + timeout
        while time.time() < deadline:
            response = self.conn.readline().decode('utf-8', errors='ignore').strip()
            if not response:
                continue
            if 'BASE_DEG' in response or 'WHEEL_DEG' in response:
                parsed = self._parse_status_line(response)
                if parsed:
                    return parsed
            elif 'ERROR' in response:
                print(f'[Serial] ⚠️  Controller error: {response}')
            # Ignore other informational lines (DONE, ZERO_OK, etc.)
        # Fallback: explicitly request latest status
        self.conn.write(b'STATUS\n')
        fallback_deadline = time.time() + 1.0
        while time.time() < fallback_deadline:
            response = self.conn.readline().decode('utf-8', errors='ignore').strip()
            if not response:
                continue
            parsed = self._parse_status_line(response)
            if parsed:
                return parsed
        return None

    def _apply_wheel_estimate(self, status, expected=None):
        data = status if status is not None else {}
        wheel_val = data.get('wheel_deg') if data else None

        if wheel_val is None and expected is None:
            return data if data else None

        if wheel_val is None:
            wheel_val = expected
        elif expected is not None and abs(wheel_val - expected) > WHEEL_STATUS_TOLERANCE:
            print(f"[Serial] ⚠️  Wheel status {wheel_val:+.2f}° disagrees with expected {expected:+.2f}°. Using expected.")
            wheel_val = expected

        if wheel_val is not None:
            if data is None:
                data = {}
            data['wheel_deg'] = wheel_val
            self.estimated_wheel_deg = wheel_val

        return data if data else None
    
    def move(self, wheel_deg):
        """Move wheel and wait for completion acknowledgement"""
        wheel_deg = int(round(wheel_deg))
        if wheel_deg == 0:
            return None
        expected = self.estimated_wheel_deg + wheel_deg
        with self.command_lock:
            with self.lock:
                cmd = f"B0 W{wheel_deg}\n"
                self.conn.write(cmd.encode('utf-8'))
                status = self._wait_for_motion_complete_locked()
        status = self._apply_wheel_estimate(status, expected)
        if status is None:
            status = {'wheel_deg': self.estimated_wheel_deg}
        return status
    
    def get_status(self):
        """Get full status including hall sensor state"""
        with self.lock:
            # Try a few times in case controller is busy
            for attempt in range(3):
                # Request status and wait for line (Serial timeout=0.5s)
                self.conn.write(b'STATUS\n')
                response = self.conn.readline().decode('utf-8', errors='ignore').strip()
                if not response:
                    continue
                parsed = self._parse_status_line(response)
                if parsed:
                    return self._apply_wheel_estimate(parsed)
            return None
    
    def get_position(self, refresh=True):
        """Get current WHEEL position"""
        status = self.get_status() if refresh else None
        if status and status.get('wheel_deg') is not None:
            return status['wheel_deg']
        return self.estimated_wheel_deg
    
    def zero(self):
        """Zero encoders"""
        with self.command_lock:
            with self.lock:
                self.conn.write(b'ZERO\n')
                status = self._wait_for_motion_complete_locked()
                self._apply_wheel_estimate(status, expected=0.0)
    
    def close(self):
        self.conn.close()


def refine_hall_alignment(serial_ctrl, last_move_direction):
    """Back off and re-approach with smaller steps for precise homing."""
    approach_dir = 1 if last_move_direction >= 0 else -1
    fine_steps = [5, 2, 1]
    try:
        print('[Homing] Refining approach with smaller steps...')
        # Step away from the hall sensor before re-approaching
        serial_ctrl.move(-approach_dir * 5)
        time.sleep(HOMING_POST_MOVE_DELAY)

        for step in fine_steps:
            serial_ctrl.move(approach_dir * step)
            time.sleep(HOMING_POST_MOVE_DELAY)
            status = serial_ctrl.get_status()
            if status and status.get('hall_wheel', 0) == 1:
                print(f'[Homing] ✅ Hall sensor confirmed with {step}° step')
                return True
        print('[Homing] ⚠️  Fine approach could not confirm hall sensor')
    except Exception as exc:
        print(f'[Homing] ⚠️  Fine approach error: {exc}')
    return False


def find_hall_home(serial_ctrl):
    """Find WHEEL home position using Hall sensor with zigzag increments 10→80°"""
    print('\n[Homing] Starting Hall sensor homing sequence...')
    
    # First, check if we're already at Hall sensor
    status = serial_ctrl.get_status()
    if status and status.get('hall_wheel', 0) == 1:  # 1 = triggered (at home)
        print('[Homing] ⚠️  Already at Hall sensor position!')
        print('[Homing] Moving away first to find edge...')
        # Move away from Hall sensor
        serial_ctrl.move(20)  # Move 20° away
        time.sleep(HOMING_POST_MOVE_DELAY)
        print('[Homing] Moved away, now searching for Hall edge...')
    
    print('[Homing] Zigzag pattern: 10° steps up to 80°')
    increments = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    direction = -1  # Start by moving downward
    last_move_dir = 0

    for idx, step in enumerate(increments, start=1):
        status = serial_ctrl.get_status()
        if status is None:
            print('[Homing] ⚠️  No status response from controller')
            time.sleep(HOMING_STATUS_DELAY)
            continue

        hall_wheel = status.get('hall_wheel', 0)
        if hall_wheel == 1:  # 1 = triggered (found home)
            print('[Homing] ✅ Hall sensor detected during sweep')
            approach_dir = last_move_dir if last_move_dir != 0 else -direction
            refine_hall_alignment(serial_ctrl, approach_dir)
            print('[Homing] ✅ Hall sensor locked. Zeroing encoders...')
            serial_ctrl.zero()
            time.sleep(0.2)
            print('[Homing] ✅ Encoders zeroed at home position')
            time.sleep(HOMING_SETTLE_DELAY)  # Hold at home for stability
            return True

        move = direction * step
        print(f'[Homing] Step {idx}/{len(increments)}: Moving {move:+d}° (Hall={hall_wheel})')
        serial_ctrl.move(move)
        time.sleep(HOMING_POST_MOVE_DELAY)
        last_move_dir = 1 if move > 0 else -1

        direction *= -1  # Alternate direction for zigzag

    print('[Homing] ⚠️  Hall sensor not found within 80° sweep')
    print('[Homing] Zeroing at current position as fallback...')
    serial_ctrl.zero()
    time.sleep(0.2)
    time.sleep(HOMING_SETTLE_DELAY)
    return False


def arducam_process(frame_queue, face_y_queue, stop_flag):
    """Isolated Arducam process"""
    print('[Arducam Process] Starting...')
    
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (ARDUCAM_WIDTH, ARDUCAM_HEIGHT)}
        )
        picam2.configure(config)
        picam2.start()
        picam2.set_controls({"AfMode": 2, "AfTrigger": 0, "FrameRate": 25.0})
        time.sleep(1)
        print('[Arducam] ✅ Camera started @ 25 FPS')
        
        # Load face detector
        try:
            face_detector = cv2.FaceDetectorYN.create(
                FACE_MODEL_PATH, "", (ARDUCAM_WIDTH, ARDUCAM_HEIGHT),
                score_threshold=0.5, nms_threshold=0.3
            )
            print('[Arducam] ✅ Face detector loaded')
        except Exception as e:
            print(f'[Arducam] ⚠️  Face detector failed: {e}')
            face_detector = None
        
        frame_time = 1.0 / 25.0  # 25 FPS limit
        last_frame_time = time.time()
        
        while not stop_flag.value:
            try:
                # FPS limiting
                elapsed = time.time() - last_frame_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last_frame_time = time.time()
                
                # Capture
                arducam_rgb = picam2.capture_array()
                if arducam_rgb is None:
                    continue
                
                arducam_bgr = cv2.cvtColor(arducam_rgb, cv2.COLOR_RGB2BGR)
                
                # Detect face
                face_y_norm = None
                if face_detector is not None:
                    _, faces = face_detector.detect(arducam_bgr)
                    if faces is not None and len(faces) > 0:
                        best_idx = int(np.argmax(faces[:, 4]))
                        y = int(faces[best_idx][1])
                        h_box = int(faces[best_idx][3])
                        cy = y + h_box // 2
                        face_y_norm = cy / ARDUCAM_HEIGHT
                
                # Resize for display
                arducam_disp = cv2.resize(arducam_bgr, (ARDUCAM_DISPLAY_WIDTH, ARDUCAM_DISPLAY_HEIGHT))
                
                # Send frame (non-blocking)
                if not frame_queue.full():
                    frame_queue.put(arducam_disp, block=False)
                
                # Send face Y (non-blocking)
                if not face_y_queue.full():
                    face_y_queue.put(face_y_norm, block=False)
                
            except Exception as e:
                print(f'[Arducam] Error: {e}')
                time.sleep(0.1)
        
        picam2.stop()
        print('[Arducam Process] Stopped')
        
    except Exception as e:
        print(f'[Arducam Process] Fatal error: {e}')


def zoomcam_process(frame_queue, stop_flag):
    """Isolated ZoomCam process with FFmpeg - using working method from track_dual_independent.py"""
    print('[ZoomCam Process] Starting...')
    
    try:
        # FFmpeg command from working tracking script
        ffmpeg_cmd = [
            "ffmpeg", "-fflags", "nobuffer", "-flags", "low_delay",
            "-i", f"udp://@:{UDP_PORT}", "-an", "-c:v", "mjpeg",
            "-q:v", "5", "-vf", f"scale={ZOOMCAM_WIDTH}:{ZOOMCAM_HEIGHT},fps=25",
            "-f", "image2pipe", "-"
        ]
        
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**6
        )
        
        if proc.stdout is None:
            raise RuntimeError("FFmpeg stdout unavailable")
        
        try:
            os.set_blocking(proc.stdout.fileno(), False)
        except:
            pass
        
        print('[ZoomCam] ✅ FFmpeg started @ 25 FPS')
        
        # Multi-threaded receiver (from working script)
        jpeg_queue = ThreadQueue(maxsize=2)
        local_frame_queue = ThreadQueue(maxsize=2)
        stream_stop = threading.Event()
        
        def reader_thread():
            buf = bytearray()
            try:
                while not stream_stop.is_set() and not stop_flag.value:
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
                while not stream_stop.is_set() and not stop_flag.value:
                    try:
                        jpeg = jpeg_queue.get(timeout=0.5)
                    except Empty:
                        continue
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    try:
                        if local_frame_queue.full():
                            try:
                                local_frame_queue.get_nowait()
                            except:
                                pass
                        local_frame_queue.put_nowait(img)
                    except:
                        pass
            finally:
                stream_stop.set()
        
        reader = threading.Thread(target=reader_thread, daemon=True)
        decoder = threading.Thread(target=decoder_thread, daemon=True)
        reader.start()
        decoder.start()
        
        # Forward frames to multiprocessing queue
        while not stop_flag.value:
            try:
                img = local_frame_queue.get(timeout=1.0)
            except Empty:
                continue
            
            # Resize for display
            zoom_disp = cv2.resize(img, (ARDUCAM_DISPLAY_WIDTH, ARDUCAM_DISPLAY_HEIGHT))
            
            # Send to main process
            if not frame_queue.full():
                try:
                    frame_queue.put(zoom_disp, block=False)
                except:
                    pass
        
        stream_stop.set()
        proc.kill()
        print('[ZoomCam Process] Stopped')
        
    except Exception as e:
        print(f'[ZoomCam Process] Fatal error: {e}')


def run_manual_calibration():
    """Manual calibration with isolated camera processes"""
    print('=' * 80)
    print('MANUAL VERTICAL CALIBRATION - ISOLATED PROCESSES')
    print('=' * 80)
    print()
    print('Controls:')
    print('  ↑         = Move WHEEL UP 1°')
    print('  ↓         = Move WHEEL DOWN 1°')
    print('  w         = Move WHEEL UP 5°')
    print('  x         = Move WHEEL DOWN 5°')
    print('  ENTER     = RECORD current position')
    print('  r         = Return to home (0°)')
    print('  s         = SAVE and quit')
    print('  q         = Quit')
    print()
    print('Strategy:')
    print('  1. System finds Hall sensor home position first')
    print('  2. Move WHEEL to different positions using arrow keys')
    print('  3. When ZoomCam points where you want, press ENTER')
    print('  4. System records where it points in Arducam')
    print('  5. Repeat for multiple positions (top, middle, bottom)')
    print('  6. Press "s" to save mapping')
    print('=' * 80)
    print()
    
    # Initialize serial FIRST (before cameras)
    try:
        serial_ctrl = FastSerial(SERIAL_PORT, BAUD_RATE)
    except Exception as e:
        print(f'❌ Serial failed: {e}')
        return
    
    # Find home position using Hall sensor BEFORE starting cameras
    print('[Setup] Finding home position with Hall sensor...')
    home_found = find_hall_home(serial_ctrl)
    
    if home_found:
        print('[Setup] ✅ Home position found and zeroed at 0°')
    else:
        print('[Setup] ⚠️  Home not found, using current position as 0°')
    
    time.sleep(0.5)
    print()
    
    # NOW start camera processes (after homing is complete)
    # Create queues and flags
    arducam_frame_queue = Queue(maxsize=2)
    arducam_face_queue = Queue(maxsize=2)
    zoomcam_frame_queue = Queue(maxsize=2)
    stop_flag = Value('i', 0)
    
    # Start camera processes
    print('[Setup] Starting camera processes...')
    arducam_proc = Process(target=arducam_process, 
                          args=(arducam_frame_queue, arducam_face_queue, stop_flag),
                          daemon=True)
    zoomcam_proc = Process(target=zoomcam_process,
                          args=(zoomcam_frame_queue, stop_flag),
                          daemon=True)
    
    arducam_proc.start()
    zoomcam_proc.start()
    
    time.sleep(2)
    print('✅ Camera processes started')
    
    # Create window
    window = 'Manual Calibration - Use Arrow Keys'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    
    # State
    mapping_points = []
    current_wheel = 0.0
    
    print()
    print('✅ Ready! Use arrow keys to move WHEEL')
    print('   Press ENTER to record positions')
    print()
    
    try:
        frame_count = 0
        fps_start = time.time()
        current_fps = 0
        
        arducam_frame = None
        zoom_frame = None
        face_y_norm = None
        
        while True:
            # Get Arducam frame
            try:
                while not arducam_frame_queue.empty():
                    arducam_frame = arducam_frame_queue.get_nowait()
            except:
                pass
            
            # Get face Y
            try:
                while not arducam_face_queue.empty():
                    face_y_norm = arducam_face_queue.get_nowait()
            except:
                pass
            
            # Get ZoomCam frame
            try:
                while not zoomcam_frame_queue.empty():
                    zoom_frame = zoomcam_frame_queue.get_nowait()
            except:
                pass
            
            # Create default frames if not available
            if arducam_frame is None:
                arducam_frame = np.zeros((ARDUCAM_DISPLAY_HEIGHT, ARDUCAM_DISPLAY_WIDTH, 3), dtype=np.uint8)
                cv2.putText(arducam_frame, 'Arducam...', (200, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            
            if zoom_frame is None:
                zoom_frame = np.zeros((ARDUCAM_DISPLAY_HEIGHT, ARDUCAM_DISPLAY_WIDTH, 3), dtype=np.uint8)
                cv2.putText(zoom_frame, 'ZoomCam...', (200, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            
            # Make copies for display
            arducam_disp = arducam_frame.copy()
            zoom_disp = zoom_frame.copy()
            
            # Visualize face detection
            if face_y_norm is not None:
                face_y_px = int(face_y_norm * ARDUCAM_DISPLAY_HEIGHT)
                cv2.line(arducam_disp, (0, face_y_px), (ARDUCAM_DISPLAY_WIDTH, face_y_px),
                        (0, 255, 0), 2)
                cv2.circle(arducam_disp, (ARDUCAM_DISPLAY_WIDTH//2, face_y_px), 8, (0, 255, 0), -1)
                cv2.putText(arducam_disp, f'Y: {face_y_norm:.3f}', 
                           (ARDUCAM_DISPLAY_WIDTH//2 + 15, face_y_px),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw ZoomCam crosshair
            cv2.line(zoom_disp, (ARDUCAM_DISPLAY_WIDTH//2 - 25, ARDUCAM_DISPLAY_HEIGHT//2),
                    (ARDUCAM_DISPLAY_WIDTH//2 + 25, ARDUCAM_DISPLAY_HEIGHT//2), (0, 255, 255), 2)
            cv2.line(zoom_disp, (ARDUCAM_DISPLAY_WIDTH//2, ARDUCAM_DISPLAY_HEIGHT//2 - 25),
                    (ARDUCAM_DISPLAY_WIDTH//2, ARDUCAM_DISPLAY_HEIGHT//2 + 25), (0, 255, 255), 2)
            
            # Combine displays
            combined = np.hstack([arducam_disp, zoom_disp])
            
            # FPS calculation
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                current_fps = frame_count
                frame_count = 0
                fps_start = time.time()
            
            # UI overlay
            cv2.putText(combined, f'ARDUCAM @ {current_fps} FPS', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined, f'ZOOMCAM @ {current_wheel:+.1f}deg', 
                       (ARDUCAM_DISPLAY_WIDTH + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Status
            status_y = ARDUCAM_DISPLAY_HEIGHT - 110
            cv2.putText(combined, f'WHEEL Position: {current_wheel:+.1f}deg', (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, f'Recorded points: {len(mapping_points)}', (10, status_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if face_y_norm is not None:
                cv2.putText(combined, f'Face detected at Y={face_y_norm:.3f}', (10, status_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(combined, 'Press ENTER to record', (10, status_y + 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(combined, 'No face detected', (10, status_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
            
            cv2.imshow(window, combined)
            
            # Handle keyboard with minimal delay
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print('\n[Exit] Quitting...')
                break
            
            elif key == ord('s'):
                if len(mapping_points) >= 2:
                    save_mapping(mapping_points)
                    break
                else:
                    print('⚠️  Need at least 2 points to save')
            
            elif key == ord('r'):
                # Return to home
                move = -current_wheel
                if abs(move) > 0.5:
                    print(f'[Move] Returning to home ({move:+.1f}°)...')
                    status = serial_ctrl.move(int(move))
                    time.sleep(0.1)
                    if status and status.get('wheel_deg') is not None:
                        current_wheel = status['wheel_deg']
                    else:
                        current_wheel = 0.0
            
            elif key == 13:  # ENTER
                if face_y_norm is not None:
                    # Get actual position from encoder
                    actual_pos = serial_ctrl.get_position(refresh=False)
                    if actual_pos is not None:
                        current_wheel = actual_pos
                    
                    mapping_points.append({
                        'wheel_degrees': current_wheel,
                        'arducam_y_normalized': face_y_norm,
                        'arducam_y_pixels': int(face_y_norm * ARDUCAM_HEIGHT)
                    })
                    
                    print(f'[Record] ✅ Point {len(mapping_points)}: WHEEL={current_wheel:+.1f}° → Y={face_y_norm:.3f}')
                    last_recorded_wheel = current_wheel
                else:
                    print('[Record] ⚠️  No face detected - cannot record')
            
            elif key == 82:  # UP arrow (move wheel upward)
                status = serial_ctrl.move(-1)
                time.sleep(0.05)
                if status and status.get('wheel_deg') is not None:
                    current_wheel = status['wheel_deg']
                else:
                    current_wheel -= 1
                print(f'[Move] ↑ {current_wheel:+.1f}°', end='\r')
            
            elif key == 84:  # DOWN arrow
                status = serial_ctrl.move(1)
                time.sleep(0.05)
                if status and status.get('wheel_deg') is not None:
                    current_wheel = status['wheel_deg']
                else:
                    current_wheel += 1
                print(f'[Move] ↓ {current_wheel:+.1f}°', end='\r')
            
            elif key == ord('w') or key == ord('W'):
                status = serial_ctrl.move(-5)
                time.sleep(0.1)
                if status and status.get('wheel_deg') is not None:
                    current_wheel = status['wheel_deg']
                else:
                    current_wheel -= 5
                print(f'[Move] ⇈ {current_wheel:+.1f}°')
            
            elif key == ord('x') or key == ord('X'):
                status = serial_ctrl.move(5)
                time.sleep(0.1)
                if status and status.get('wheel_deg') is not None:
                    current_wheel = status['wheel_deg']
                else:
                    current_wheel += 5
                print(f'[Move] ⇊ {current_wheel:+.1f}°')
    
    finally:
        # Stop processes
        stop_flag.value = 1
        time.sleep(0.5)
        
        if arducam_proc.is_alive():
            arducam_proc.terminate()
        if zoomcam_proc.is_alive():
            zoomcam_proc.terminate()
        
        arducam_proc.join(timeout=1)
        zoomcam_proc.join(timeout=1)
        
        serial_ctrl.close()
        cv2.destroyAllWindows()
        print('\n✅ Cleanup complete')


def save_mapping(mapping_points):
    """Save mapping to JSON"""
    # Sort by wheel degrees
    mapping_points = sorted(mapping_points, key=lambda p: p['wheel_degrees'])
    
    wheel_min = min(p['wheel_degrees'] for p in mapping_points)
    wheel_max = max(p['wheel_degrees'] for p in mapping_points)
    
    data = {
        'timestamp': time.time(),
        'mapping_points': mapping_points,
        'wheel_limits': {
            'min_degrees': wheel_min,
            'max_degrees': wheel_max,
            'range_degrees': wheel_max - wheel_min
        },
        'num_points': len(mapping_points),
        'calibration_version': '3.0_manual_realtime'
    }
    
    with open(MAPPING_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f'\n[Save] ✅ Saved {len(mapping_points)} points to {MAPPING_FILE}')
    print(f'[Save] WHEEL range: {wheel_min:.1f}° to {wheel_max:.1f}°')
    print('[Save] Mapping points:')
    for i, p in enumerate(mapping_points, 1):
        print(f'  {i}. WHEEL={p["wheel_degrees"]:+6.1f}° → Y={p["arducam_y_normalized"]:.3f}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    run_manual_calibration()
