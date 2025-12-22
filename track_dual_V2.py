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
import json
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
WHEEL_TARGET_TOLERANCE = 1.5  # Degrees within which we consider the wheel on target - INCREASED to prevent oscillation
WHEEL_MIN_MOVE_THRESHOLD = 2.0  # Minimum error (degrees) before making ANY correction - prevents tiny adjustments
WHEEL_TARGET_CHANGE_MIN = 0.5  # Ignore target changes smaller than this (degrees) - debouncing
WHEEL_MOVE_COOLDOWN = 0.8  # Minimum seconds between wheel commands - prevents rapid oscillation

# Vertical calibration mapping
VERTICAL_MAPPING_FILE = 'vertical_mapping.json'

# Homing and serial timing
HOMING_STATUS_DELAY = 0.5
HOMING_POST_MOVE_DELAY = 0.8
HOMING_SETTLE_DELAY = 2.0
HOMING_SWEEP_STEPS = [10, 20, 30, 40, 50, 60, 70, 80]
BASE_STEPWISE_INCREMENT = 10  # Degrees per incremental base homing command
WHEEL_STEPWISE_INCREMENT = 5  # Degrees per incremental wheel homing command
BASE_HALL_FINE_INCREMENT = 1  # Fine step (°) when monitoring hall sensor on base
WHEEL_HALL_FINE_INCREMENT = 1  # Fine step (°) when monitoring hall sensor on wheel
STEPWISE_COMMAND_DELAY = 0.01  # Delay between coarse incremental commands
HALL_MONITOR_DELAY = 0.0  # Delay between fine hall-monitoring steps
MOTION_COMPLETE_TIMEOUT = 6.0  # Seconds to wait for controller status after a move
WHEEL_STATUS_TOLERANCE = 0.75  # Degrees before trusting cached estimate over status

# Variable distance handling (for blur/close objects)
BLUR_VARIANCE_THRESHOLD = 80.0  # Laplacian variance threshold - below this = blurry
BLUR_SMOOTHING_FACTOR = 0.85  # Extra smoothing when blur detected (higher = more stable)
FACE_SIZE_CLOSE_THRESHOLD = 0.35  # Normalized face width threshold for "close" detection
FACE_SIZE_FAR_THRESHOLD = 0.15  # Normalized face width threshold for "far" detection
CLOSE_DISTANCE_SCORE_THRESHOLD = 0.4  # Lower detection threshold when object is close and blurry
TEMPORAL_PREDICTION_FRAMES = 5  # Number of past positions to use for motion prediction
PREDICTION_TIMEOUT = 1.0  # Seconds to continue prediction after losing face
MAPPING_TRUST_WHEN_BLURRY = 0.9  # Weighting factor: rely more on Arducam mapping when ZoomCam is blurry
# ============================


# ========== SHARED SERIAL CONNECTION ==========
# Global serial object and lock (shared across processes via Manager)
serial_conn = None
serial_lock = None
shared_face_state = None  # Shared dict for face position comparison
shared_motion_state = None  # Shared wheel/base estimates across processes


class VerticalMapping:
    """Linear interpolation helper from Arducam Y to wheel degrees."""
    def __init__(self, samples):
        if not samples:
            raise ValueError('Vertical mapping requires at least one sample')
        cleaned = []
        for item in samples:
            y = float(item['arducam_y_normalized'])
            w = float(item['wheel_degrees'])
            cleaned.append((y, w))
        cleaned.sort(key=lambda p: p[0])
        # Remove duplicates with identical Y by keeping the latest entry
        unique = []
        for y, w in cleaned:
            if not unique or abs(y - unique[-1][0]) > 1e-6:
                unique.append((y, w))
            else:
                unique[-1] = (y, w)
        self.samples = unique
        self.y_min = self.samples[0][0]
        self.y_max = self.samples[-1][0]

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        points = data.get('mapping_points', [])
        if not points:
            raise ValueError('Mapping file is missing "mapping_points" entries')
        return cls(points)

    def lookup(self, y_norm):
        if not self.samples:
            return 0.0
        y = min(max(y_norm, self.y_min), self.y_max)
        if y <= self.samples[0][0]:
            return self.samples[0][1]
        for i in range(1, len(self.samples)):
            y0, w0 = self.samples[i-1]
            y1, w1 = self.samples[i]
            if y <= y1:
                if abs(y1 - y0) < 1e-6:
                    return w1
                t = (y - y0) / (y1 - y0)
                return w0 + t * (w1 - w0)
        return self.samples[-1][1]


def parse_status_line(line):
    if 'BASE_DEG' not in line and 'WHEEL_DEG' not in line:
        return None
    result = {}
    for part in line.split('|'):
        part = part.strip()
        if '=' not in part:
            continue
        key, val = part.split('=', 1)
        key = key.strip()
        val = val.strip()
        try:
            if key in ('BASE_DEG', 'WHEEL_DEG'):
                result[key.lower()] = float(val)
            elif key in ('HALL_BASE', 'HALL_WHEEL'):
                result[key.lower()] = int(val)
        except ValueError:
            continue
    return result if result else None


def attach_motion_state(state):
    """Assign the Manager-backed motion state dict to this process."""
    global shared_motion_state
    shared_motion_state = state


def get_shared_wheel_deg(default=0.0):
    if shared_motion_state is None:
        return default
    try:
        return float(shared_motion_state.get('wheel_deg', default))
    except Exception:
        return default


def set_shared_wheel_deg(value):
    if shared_motion_state is None:
        return
    try:
        shared_motion_state['wheel_deg'] = float(value)
    except Exception:
        pass


def apply_wheel_estimate(status, expected=None):
    data = status.copy() if status else {}
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
        set_shared_wheel_deg(wheel_val)

    return data if data else None


def wait_for_motion_complete_unlocked(expected=None):
    """Wait for controller to print a status line after a move (lock must be held)."""
    if serial_conn is None:
        return None

    deadline = time.time() + MOTION_COMPLETE_TIMEOUT
    while time.time() < deadline:
        response = serial_conn.readline().decode('utf-8', errors='ignore').strip()
        if not response:
            continue
        if 'BASE_DEG' in response or 'WHEEL_DEG' in response:
            parsed = parse_status_line(response)
            if parsed:
                return apply_wheel_estimate(parsed, expected)
        elif 'ERROR' in response:
            print(f'[Serial] ⚠️  Controller error: {response}')

    # Fallback request
    serial_conn.write(b'STATUS\n')
    fallback_deadline = time.time() + 1.0
    while time.time() < fallback_deadline:
        response = serial_conn.readline().decode('utf-8', errors='ignore').strip()
        if not response:
            continue
        parsed = parse_status_line(response)
        if parsed:
            return apply_wheel_estimate(parsed, expected)
    return None


def request_status(serial_lock, attempts=3):
    global serial_conn
    if serial_conn is None:
        return None
    for _ in range(attempts):
        try:
            with serial_lock:
                serial_conn.write(b'STATUS\n')
                response = serial_conn.readline().decode('utf-8', errors='ignore').strip()
        except Exception as exc:
            print(f'[Serial] STATUS error: {exc}')
            continue
        if not response:
            time.sleep(HOMING_STATUS_DELAY)
            continue
        parsed = parse_status_line(response)
        if parsed:
            return apply_wheel_estimate(parsed)
    return None


def send_wheel_delta(delta_deg, serial_lock):
    global serial_conn
    delta = int(round(delta_deg))
    if delta == 0 or serial_conn is None:
        return None
    expected = get_shared_wheel_deg() + delta
    try:
        with serial_lock:
            cmd = f"B0 W{delta}\n"
            serial_conn.write(cmd.encode('utf-8'))
            status = wait_for_motion_complete_unlocked(expected)
            return status
    except Exception as exc:
        print(f'[Serial] Move error: {exc}')
    return None


def send_base_delta(delta_deg, serial_lock):
    global serial_conn
    delta = int(round(delta_deg))
    if delta == 0 or serial_conn is None:
        return None
    expected_wheel = get_shared_wheel_deg()
    try:
        with serial_lock:
            cmd = f"B{delta} W0\n"
            serial_conn.write(cmd.encode('utf-8'))
            status = wait_for_motion_complete_unlocked(expected=expected_wheel)
            return status
    except Exception as exc:
        print(f'[Serial] Base move error: {exc}')
    return None


def zero_encoders(serial_lock):
    global serial_conn
    if serial_conn is None:
        return
    try:
        with serial_lock:
            serial_conn.write(b'ZERO\n')
            wait_for_motion_complete_unlocked(expected=0.0)
            set_shared_wheel_deg(0.0)
    except Exception as exc:
        print(f'[Serial] ZERO error: {exc}')


def move_wheel_stepwise(serial_lock, total_degrees, stop_on_hall=False):
    """Move the wheel axis in incremental steps while optionally monitoring hall events mid-sweep.

    Returns (hall_triggered, last_direction, last_status).
    """
    magnitude = abs(total_degrees)
    if magnitude == 0:
        return False, 0, None

    direction = 1 if total_degrees > 0 else -1
    step_increment = max(1, WHEEL_STEPWISE_INCREMENT)
    delay_between_steps = STEPWISE_COMMAND_DELAY

    if stop_on_hall:
        fine_increment = max(1, WHEEL_HALL_FINE_INCREMENT)
        step_increment = min(step_increment, fine_increment)
        delay_between_steps = HALL_MONITOR_DELAY

    steps = int(magnitude // step_increment)
    remainder = magnitude % step_increment
    hall_triggered = False
    last_status = None

    # Move in WHEEL_STEPWISE_INCREMENT degree chunks
    for _ in range(steps):
        status = send_wheel_delta(direction * step_increment, serial_lock)
        if status:
            last_status = status
            if stop_on_hall and status.get('hall_wheel', 0) == 1:
                hall_triggered = True
                break
        elif stop_on_hall:
            check = request_status(serial_lock)
            if check:
                last_status = check
                if check.get('hall_wheel', 0) == 1:
                    hall_triggered = True
                    break
        # Small pause keeps the command stream from overwhelming the controller
        if delay_between_steps > 0:
            time.sleep(delay_between_steps)

    # Handle remainder degrees if no hall trigger yet
    if not hall_triggered and remainder > 0:
        status = send_wheel_delta(direction * remainder, serial_lock)
        if status:
            last_status = status
            if stop_on_hall and status.get('hall_wheel', 0) == 1:
                hall_triggered = True
        elif stop_on_hall:
            check = request_status(serial_lock)
            if check:
                last_status = check
                if check.get('hall_wheel', 0) == 1:
                    hall_triggered = True

    return hall_triggered, direction, last_status


def move_base_stepwise(serial_lock, total_degrees, stop_on_hall=False):
    """Move the base axis in incremental steps while optionally monitoring hall events mid-sweep.

    Returns (hall_triggered, last_direction, last_status).
    """
    magnitude = abs(total_degrees)
    if magnitude == 0:
        return False, 0, None

    direction = 1 if total_degrees > 0 else -1
    step_increment = max(1, BASE_STEPWISE_INCREMENT)
    delay_between_steps = STEPWISE_COMMAND_DELAY

    if stop_on_hall:
        fine_increment = max(1, BASE_HALL_FINE_INCREMENT)
        step_increment = min(step_increment, fine_increment)
        delay_between_steps = HALL_MONITOR_DELAY

    steps = int(magnitude // step_increment)
    remainder = magnitude % step_increment
    hall_triggered = False
    last_status = None

    # Move in BASE_STEPWISE_INCREMENT degree chunks
    for _ in range(steps):
        status = send_base_delta(direction * step_increment, serial_lock)
        if status:
            last_status = status
            if stop_on_hall and status.get('hall_base', 0) == 1:
                hall_triggered = True
                break
        elif stop_on_hall:
            check = request_status(serial_lock)
            if check:
                last_status = check
                if check.get('hall_base', 0) == 1:
                    hall_triggered = True
                    break
        # Small pause keeps the command stream from overwhelming the controller
        if delay_between_steps > 0:
            time.sleep(delay_between_steps)

    # Handle remainder degrees if no hall trigger yet
    if not hall_triggered and remainder > 0:
        status = send_base_delta(direction * remainder, serial_lock)
        if status:
            last_status = status
            if stop_on_hall and status.get('hall_base', 0) == 1:
                hall_triggered = True
        elif stop_on_hall:
            check = request_status(serial_lock)
            if check:
                last_status = check
                if check.get('hall_base', 0) == 1:
                    hall_triggered = True

    return hall_triggered, direction, last_status


def refine_wheel_hall_alignment(serial_lock, last_move_direction):
    approach_dir = 1 if last_move_direction >= 0 else -1
    fine_steps = [5, 2, 1]
    print('[Homing][Wheel] Refining hall alignment...')
    move_wheel_stepwise(serial_lock, -approach_dir * 5, stop_on_hall=False)
    time.sleep(HOMING_POST_MOVE_DELAY)
    for step in fine_steps:
        hall_triggered, _, _ = move_wheel_stepwise(serial_lock, approach_dir * step, stop_on_hall=True)
        time.sleep(HOMING_POST_MOVE_DELAY)
        if hall_triggered:
            print(f'[Homing][Wheel] ✅ Confirmed with {step}° step (immediate stop)')
            return True
    print('[Homing][Wheel] ⚠️  Fine approach failed to confirm hall sensor')
    return False


def find_wheel_hall_home(serial_lock):
    print('\n[Homing][Wheel] Starting Hall sensor homing sequence...')
    status = request_status(serial_lock)
    if status and status.get('hall_wheel', 0) == 1:
        print('[Homing][Wheel] ⚠️  Already at Hall sensor. Moving away first...')
        send_wheel_delta(20, serial_lock)
        time.sleep(HOMING_POST_MOVE_DELAY)

    direction = 1  # Start with positive direction
    last_move_dir = 0
    for idx, step in enumerate(HOMING_SWEEP_STEPS, start=1):
        status = request_status(serial_lock)
        if status is None:
            print('[Homing][Wheel] ⚠️  No status response')
            continue
        hall = status.get('hall_wheel', 0)
        if hall == 1:
            print('[Homing][Wheel] ✅ Hall sensor detected during sweep')
            approach = last_move_dir if last_move_dir != 0 else -direction
            refine_wheel_hall_alignment(serial_lock, approach)
            zero_encoders(serial_lock)
            time.sleep(HOMING_SETTLE_DELAY)
            return True
        move = direction * step
        print(f'[Homing][Wheel] Step {idx}/{len(HOMING_SWEEP_STEPS)}: Moving {move:+d}° (monitoring hall, {WHEEL_STEPWISE_INCREMENT}° increments)')
        hall_triggered, last_move_dir, status = move_wheel_stepwise(serial_lock, move, stop_on_hall=True)
        time.sleep(HOMING_POST_MOVE_DELAY)
        if hall_triggered:
            if status:
                print(f"[Homing][Wheel] Status @ hall: BASE_DEG={status.get('base_deg', '?')} WHEEL_DEG={status.get('wheel_deg', '?')} HALL_BASE={status.get('hall_base', '?')} HALL_WHEEL={status.get('hall_wheel', '?')}")
            print('[Homing][Wheel] ✅ Hall sensor triggered mid-move — stopping immediately')
            zero_encoders(serial_lock)
            time.sleep(HOMING_SETTLE_DELAY)
            return True
        direction *= -1

    print('[Homing][Wheel] ⚠️  Hall sensor not found. Zeroing current position...')
    zero_encoders(serial_lock)
    time.sleep(HOMING_SETTLE_DELAY)
    return False


def refine_base_hall_alignment(serial_lock, last_move_direction):
    approach_dir = 1 if last_move_direction >= 0 else -1
    fine_steps = [5, 2, 1]
    print('[Homing][Base] Refining hall alignment...')
    move_base_stepwise(serial_lock, -approach_dir * 5, stop_on_hall=False)
    time.sleep(HOMING_POST_MOVE_DELAY)
    for step in fine_steps:
        hall_triggered, _, status = move_base_stepwise(serial_lock, approach_dir * step, stop_on_hall=True)
        time.sleep(HOMING_POST_MOVE_DELAY)
        if hall_triggered or (status and status.get('hall_base', 0) == 1):
            print(f'[Homing][Base] ✅ Confirmed with {step}° step (immediate stop)')
            return True
    print('[Homing][Base] ⚠️  Fine approach failed to confirm hall sensor')
    return False


def find_base_hall_home(serial_lock):
    print('\n[Homing][Base] Starting Hall sensor homing sequence...')
    status = request_status(serial_lock)
    if status and status.get('hall_base', 0) == 1:
        print('[Homing][Base] ⚠️  Already at Hall sensor. Moving away first...')
        send_base_delta(20, serial_lock)
        time.sleep(HOMING_POST_MOVE_DELAY)

    direction = -1  # Start with negative direction
    last_move_dir = 0
    for idx, step in enumerate(HOMING_SWEEP_STEPS, start=1):
        status = request_status(serial_lock)
        if status is None:
            print('[Homing][Base] ⚠️  No status response')
            continue
        hall = status.get('hall_base', 0)
        if hall == 1:
            print('[Homing][Base] ✅ Hall sensor detected during sweep')
            approach = last_move_dir if last_move_dir != 0 else -direction
            refine_base_hall_alignment(serial_lock, approach)
            time.sleep(HOMING_SETTLE_DELAY)
            return True

        move = direction * step
        print(f'[Homing][Base] Step {idx}/{len(HOMING_SWEEP_STEPS)}: Moving {move:+d}° (monitoring hall, {BASE_STEPWISE_INCREMENT}° increments)')
        hall_triggered, last_move_dir, status = move_base_stepwise(serial_lock, move, stop_on_hall=True)
        time.sleep(HOMING_POST_MOVE_DELAY)
        if hall_triggered:
            if status:
                print(f"[Homing][Base] Status @ hall: BASE_DEG={status.get('base_deg', '?')} WHEEL_DEG={status.get('wheel_deg', '?')} HALL_BASE={status.get('hall_base', '?')} HALL_WHEEL={status.get('hall_wheel', '?')}")
            print('[Homing][Base] ✅ Hall sensor triggered mid-move — stopping immediately')
            time.sleep(HOMING_SETTLE_DELAY)
            return True
        direction *= -1

    print('[Homing][Base] ⚠️  Hall sensor not found. Leaving BASE at current position.')
    return False

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


# ========== ARDUCAM PROCESS (Horizontal Tracking + Face Position Sharing) ==========
def arducam_process(serial_lock, stop_event, shared_face_state, motion_state):
    """
    Arducam Process - Horizontal (BASE) tracking + Face position sharing
    Detects face on Arducam → controls BASE + shares face Y position for vertical coordination
    
    """
    attach_motion_state(motion_state)
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
    # cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # DISABLED: No visual display
    
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
                wait_for_motion_complete_unlocked(expected=get_shared_wheel_deg())
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
                # Clear face state when no face detected
                shared_face_state['has_face'] = False
            
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
            
            # cv2.imshow(window, disp)  # DISABLED: No visual display
            
            # Still need waitKey for OpenCV processing, but no window display
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('[Arducam] Quit')
                stop_event.set()
                break
    
    finally:
        try:
            picam2.stop()
        except:
            pass
        # cv2.destroyWindow(window)  # DISABLED: No window to destroy
        print('[Arducam] Stopped')


# ========== BLUR DETECTION & TEMPORAL SMOOTHING HELPERS ==========
def calculate_blur_score(image):
    """Calculate blur score using Laplacian variance. Lower = more blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def estimate_distance_category(face_width_normalized):
    """Estimate if object is close, medium, or far based on face size."""
    if face_width_normalized > FACE_SIZE_CLOSE_THRESHOLD:
        return 'close'
    elif face_width_normalized < FACE_SIZE_FAR_THRESHOLD:
        return 'far'
    else:
        return 'medium'


class TemporalSmoother:
    """Tracks face positions over time and provides smoothed/predicted positions."""
    def __init__(self, max_history=TEMPORAL_PREDICTION_FRAMES):
        self.max_history = max_history
        self.positions = []  # List of (timestamp, y_norm, confidence)
        
    def add(self, timestamp, y_norm, confidence=1.0):
        """Add a new position measurement."""
        self.positions.append((timestamp, y_norm, confidence))
        if len(self.positions) > self.max_history:
            self.positions.pop(0)
    
    def get_smoothed(self):
        """Get weighted average of recent positions."""
        if not self.positions:
            return None
        
        # Weight recent positions more heavily
        total_weight = 0.0
        weighted_sum = 0.0
        for i, (ts, y, conf) in enumerate(self.positions):
            recency_weight = (i + 1) / len(self.positions)  # 0.2, 0.4, 0.6, 0.8, 1.0 for 5 frames
            weight = recency_weight * conf
            weighted_sum += y * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else None
    
    def predict(self, current_time):
        """Predict position based on velocity (linear extrapolation)."""
        if len(self.positions) < 2:
            return self.get_smoothed()
        
        # Use last 2 positions to estimate velocity
        t1, y1, _ = self.positions[-2]
        t2, y2, _ = self.positions[-1]
        
        dt = t2 - t1
        if dt <= 0:
            return y2
        
        velocity = (y2 - y1) / dt
        time_since_last = current_time - t2
        
        # Only predict for short time periods
        if time_since_last > PREDICTION_TIMEOUT:
            return None
        
        predicted_y = y2 + velocity * time_since_last
        # Clamp to valid range
        return max(0.0, min(1.0, predicted_y))
    
    def clear(self):
        """Clear history."""
        self.positions.clear()


# ========== ZOOMCAM PROCESS (Vertical Tracking) ==========
def zoomcam_process(serial_lock, stop_event, shared_face_state, motion_state, mapping_points, initial_wheel_deg):
    """ZoomCam process - drives WHEEL to mapped Arducam face height."""
    attach_motion_state(motion_state)
    print('[ZoomCam] Starting mapped vertical tracking...')

    try:
        vertical_mapping = VerticalMapping(mapping_points)
        print(f'[ZoomCam] ✅ Loaded mapping ({len(vertical_mapping.samples)} samples)')
    except Exception as exc:
        print(f'[ZoomCam] ❌ Mapping failed: {exc}')
        stop_event.set()
        return
    
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
    # cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # DISABLED: No visual display
    
    # === State ===
    frame_count = 0
    fps_start = time.time()
    current_fps = 0
    last_send_time = 0.0
    estimated_wheel_deg = float(get_shared_wheel_deg(initial_wheel_deg))
    
    # Temporal smoothing for variable distance handling
    temporal_smoother = TemporalSmoother(max_history=TEMPORAL_PREDICTION_FRAMES)
    last_face_time = 0.0
    smoothed_target = None
    last_commanded_target = None  # Track last target we actually moved to
    last_wheel_move_time = 0.0  # Track when we last moved the wheel

    def drive_wheel_to_target(target_deg):
        nonlocal last_send_time, estimated_wheel_deg, last_commanded_target, last_wheel_move_time
        
        # Refresh estimate from shared state in case another process updated it
        estimated_wheel_deg = get_shared_wheel_deg(estimated_wheel_deg)
        delta = target_deg - estimated_wheel_deg
        
        # ANTI-OSCILLATION CHECK 1: Already within tolerance - don't move
        if abs(delta) <= WHEEL_TARGET_TOLERANCE:
            return
        
        # ANTI-OSCILLATION CHECK 2: Error too small to bother correcting
        if abs(delta) < WHEEL_MIN_MOVE_THRESHOLD:
            return
        
        # ANTI-OSCILLATION CHECK 3: Target hasn't changed significantly since last move
        if last_commanded_target is not None:
            target_change = abs(target_deg - last_commanded_target)
            if target_change < WHEEL_TARGET_CHANGE_MIN:
                return
        
        # ANTI-OSCILLATION CHECK 4: Cooldown period - prevent rapid-fire commands
        now = time.time()
        if now - last_wheel_move_time < WHEEL_MOVE_COOLDOWN:
            return

        # Calculate step size
        step = int(np.clip(delta, -MAX_STEP_WHEEL, MAX_STEP_WHEEL))
        if step == 0:
            step = 1 if delta > 0 else -1

        # Rate limiting (additional safety)
        if now - last_send_time < SEND_PERIOD:
            return

        # Execute the move
        status = send_wheel_delta(step, serial_lock)
        last_send_time = now
        last_wheel_move_time = now
        last_commanded_target = target_deg
        
        if status and status.get('wheel_deg') is not None:
            estimated_wheel_deg = status['wheel_deg']
        else:
            estimated_wheel_deg += step
            set_shared_wheel_deg(estimated_wheel_deg)
        print(f'[ZoomCam] Target {target_deg:+.1f}°, move {step:+d}° → est {estimated_wheel_deg:+.1f}° (Δ={delta:+.1f}°)')
        time.sleep(SETTLING_DELAY)

    print('[ZoomCam] 🎯 Mapping Arducam face position to wheel angle (variable distance aware)')
    print('[ZoomCam] Processing in headless mode (no display)')
    
    # === Main Loop ===
    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1.0)
            except Empty:
                # Check for stop event (no keyboard input without display)
                time.sleep(0.01)
                continue
            
            h, w = frame.shape[:2]
            current_time = time.time()
            
            # FPS
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                current_fps = frame_count
                frame_count = 0
                fps_start = time.time()
            
            # === BLUR DETECTION ===
            blur_score = calculate_blur_score(frame)
            is_blurry = blur_score < BLUR_VARIANCE_THRESHOLD
            
            # === GET ARDUCAM FACE POSITION ===
            arducam_has_face = shared_face_state.get('has_face', False)
            arducam_face_y_norm = shared_face_state.get('arducam_face_y', 0.5)
            arducam_timestamp = shared_face_state.get('timestamp', 0)
            
            # Check if Arducam data is fresh (within last 0.5 seconds)
            data_is_fresh = (time.time() - arducam_timestamp) < 0.5
            target_wheel_deg = None
            
            # === DETECT FACE IN ZOOMCAM (with adaptive threshold for blur) ===
            # Adjust detection threshold based on blur
            if is_blurry:
                face_detector.setScoreThreshold(CLOSE_DISTANCE_SCORE_THRESHOLD)
            else:
                face_detector.setScoreThreshold(0.5)
            
            _, faces = face_detector.detect(frame)
            
            zoomcam_has_face = False
            zoomcam_face_y_norm = 0.5
            face_width_norm = 0.0
            distance_category = 'unknown'
            
            if faces is not None and len(faces) > 0:
                # Get largest face
                best_idx = int(np.argmax(faces[:, 4]))
                x, y, box_w, box_h, score = faces[best_idx, :5]
                
                x, y = int(x), int(y)
                box_w, box_h = int(box_w), int(box_h)
                cx = x + box_w // 2
                cy = y + box_h // 2
                zoomcam_has_face = True
                
                # Calculate normalized Y position and face size for ZoomCam
                zoomcam_face_y_norm = cy / h
                face_width_norm = box_w / w
                distance_category = estimate_distance_category(face_width_norm)
                
                # Update temporal smoother with current detection
                confidence = score  # Use detection score as confidence
                if is_blurry:
                    confidence *= 0.7  # Lower confidence for blurry detections
                temporal_smoother.add(current_time, zoomcam_face_y_norm, confidence)
                last_face_time = current_time
                
                # Draw face with color based on distance
                box_color = (0, 255, 0) if not is_blurry else (0, 165, 255)  # Green or orange
                cv2.rectangle(frame, (x, y), (x+box_w, y+box_h), box_color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'ZoomCam Y: {zoomcam_face_y_norm:.2f}', (cx+10, cy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, f'{distance_category} ({face_width_norm:.2f})', (cx+10, cy+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            else:
                # No face detected - try prediction from recent history
                if current_time - last_face_time < PREDICTION_TIMEOUT:
                    predicted_y = temporal_smoother.predict(current_time)
                    if predicted_y is not None:
                        zoomcam_face_y_norm = predicted_y
                        # Don't set zoomcam_has_face, but use for mapping correction
                        cv2.putText(frame, f'Predicted Y: {predicted_y:.2f}', (10, 140),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 255), 2)
            
            # === MAP ARDUCAM FACE TO WHEEL TARGET (with blur-aware blending) ===
            if arducam_has_face and data_is_fresh:
                arducam_target = vertical_mapping.lookup(arducam_face_y_norm)
                
                # If blurry, rely more heavily on Arducam mapping
                if is_blurry and zoomcam_has_face:
                    # Blend: more weight to Arducam when blurry
                    blend_factor = MAPPING_TRUST_WHEN_BLURRY
                    # Get smoothed ZoomCam position for minor correction
                    smoothed_y = temporal_smoother.get_smoothed()
                    if smoothed_y is not None:
                        zoomcam_suggested_target = vertical_mapping.lookup(smoothed_y)
                        target_wheel_deg = (blend_factor * arducam_target + 
                                          (1 - blend_factor) * zoomcam_suggested_target)
                    else:
                        target_wheel_deg = arducam_target
                else:
                    # Normal operation: trust Arducam mapping
                    target_wheel_deg = arducam_target
                
                # Apply smoothing for stability - HEAVY smoothing to prevent oscillation
                if smoothed_target is None:
                    smoothed_target = target_wheel_deg
                else:
                    # More smoothing when blurry, but always use heavy smoothing to prevent oscillation
                    alpha = BLUR_SMOOTHING_FACTOR if is_blurry else 0.85  # Increased from 0.7 to 0.85
                    smoothed_target = alpha * smoothed_target + (1 - alpha) * target_wheel_deg
                
                drive_wheel_to_target(smoothed_target)
                
                # Draw Arducam reference line
                arducam_y_in_frame = int(arducam_face_y_norm * h)
                cv2.line(frame, (0, arducam_y_in_frame), (w, arducam_y_in_frame), (255, 128, 0), 2)
                cv2.putText(frame, f'Arducam Y: {arducam_face_y_norm:.2f}', (10, arducam_y_in_frame-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
                if smoothed_target is not None:
                    cv2.putText(frame, f'Target wheel: {smoothed_target:+.1f}°', (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                               
            estimated_wheel_deg = get_shared_wheel_deg(estimated_wheel_deg)
            cv2.putText(frame, f'Estimated wheel: {estimated_wheel_deg:+.1f}°', (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)
            
            # UI overlay with blur indicator
            title_color = (0, 255, 255) if not is_blurry else (0, 165, 255)
            cv2.putText(frame, 'ZOOMCAM - Variable Distance Tracking', (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2)
            cv2.putText(frame, f'{w}x{h} @ {current_fps} FPS', (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f'Blur: {blur_score:.1f} {"BLURRY" if is_blurry else "SHARP"}', (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255) if is_blurry else (0, 255, 0), 2)
            
            # Status based on face detection in both cameras and distance
            if arducam_has_face and data_is_fresh:
                if is_blurry:
                    status = f'Mapping (blur-compensated, {distance_category})'
                else:
                    status = f'Mapping control ({distance_category})'
            elif zoomcam_has_face:
                status = f'ZoomCam only ({distance_category})'
            else:
                if current_time - last_face_time < PREDICTION_TIMEOUT:
                    status = 'Predicting from history'
                else:
                    status = 'No faces detected'
            
            cv2.putText(frame, f'Mode: {status}', (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Center horizontal line
            cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 1)
            
            # cv2.imshow(window, frame)  # DISABLED: No visual display
            
            # Still need waitKey for OpenCV processing, but no window display
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('[ZoomCam] Quit')
                stop_event.set()
                break
    
    finally:
        stream_stop.set()
        try:
            proc.kill()
        except:
            pass
        # cv2.destroyWindow(window)  # DISABLED: No window to destroy
        print('[ZoomCam] Stopped')


# ========== MAIN ==========
def main():
    global serial_conn, serial_lock, shared_face_state, shared_motion_state
    
    print('=' * 80)
    print('DUAL-CAMERA COORDINATED TRACKING - HEADLESS MODE')
    print('=' * 80)
    print()
    print('🎯 Architecture:')
    print('  Process 1: Arducam → Detects face → Controls BASE (horizontal) + shares face Y')
    print('  Process 2: ZoomCam → Maps Arducam face Y → Drives WHEEL (vertical)')
    print()
    print('�️  Headless mode: No visual display windows')
    print('  - All face detection and tracking active')
    print('  - Motor control fully functional')
    print('  - Monitoring via terminal output only')
    print()
    print('✅ Vertical control uses recorded calibration mapping + homing before start')
    print('⚠️  Use Ctrl+C to stop all processes.')
    print('=' * 80)
    print()
    
    # Initialize serial connection ONCE
    if not init_serial():
        print('❌ Failed to initialize serial. Exiting.')
        return
    
    # Create shared lock/state and load mapping
    manager = Manager()
    serial_lock = manager.Lock()
    shared_face_state = manager.dict()
    shared_face_state['has_face'] = False
    shared_face_state['arducam_face_y'] = 0.5
    shared_face_state['timestamp'] = 0
    shared_motion_state = manager.dict()
    shared_motion_state['wheel_deg'] = 0.0

    try:
        mapping = VerticalMapping.load(VERTICAL_MAPPING_FILE)
        mapping_payload = [
            {'arducam_y_normalized': y, 'wheel_degrees': w}
            for y, w in mapping.samples
        ]
        print(f'[Setup] ✅ Loaded {len(mapping.samples)} mapping samples from {VERTICAL_MAPPING_FILE}')
    except Exception as exc:
        print(f'[Setup] ❌ Failed to load mapping: {exc}')
        return

    base_home_found = find_base_hall_home(serial_lock)
    if base_home_found:
        print('[Setup] ✅ Base homed to hall sensor')
    else:
        print('[Setup] ⚠️  Base homing failed — continuing from current position')

    wheel_home_found = find_wheel_hall_home(serial_lock)
    if wheel_home_found:
        print('[Setup] ✅ Wheel homed and zeroed')
    else:
        print('[Setup] ⚠️  Wheel homing fallback used (zeroed at current position)')
    set_shared_wheel_deg(0.0)
    initial_wheel_deg = 0.0
    
    stop_event = MPEvent()
    
    # Start both camera processes with shared state
    p_arducam = Process(target=arducam_process,
                        args=(serial_lock, stop_event, shared_face_state, shared_motion_state),
                        name='Arducam')
    p_zoomcam = Process(
        target=zoomcam_process,
        args=(serial_lock, stop_event, shared_face_state, shared_motion_state, mapping_payload, initial_wheel_deg),
        name='ZoomCam'
    )
    
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