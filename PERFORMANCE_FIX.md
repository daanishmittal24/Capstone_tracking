# ZoomCam Performance Fix

## Problem
ZoomCam running at **1 FPS** causing:
- Face detected → motor moves → face leaves frame → re-detect loop
- Unusable preview and jerky tracking

## Root Cause
```
Main Loop (1 FPS):
┌──────────────────────────────────┐
│ 1. Get frame         (~50ms)     │
│ 2. Detect face       (~100ms)    │
│ 3. Calculate target  (~1ms)      │
│ 4. Send motor cmd    (~100ms)    │ ← BLOCKING serial I/O
│ 5. time.sleep(0.35s) (350ms)     │ ← BLOCKING settling delay
│ 6. Render overlay    (~10ms)     │
└──────────────────────────────────┘
Total: ~611ms/frame = 1.6 FPS
```

The `drive_wheel_to_target()` function was:
1. Sending serial commands **synchronously** (blocks 100ms)
2. Waiting for motion complete status (blocks 100ms)
3. Sleeping for settling delay (blocks 350ms)
4. **Blocking the entire frame processing loop**

## Solution

### 1. Asynchronous Motor Control Thread
```
Main Loop (20 FPS):              Motor Thread (background):
┌─────────────────────┐           ┌──────────────────────┐
│ 1. Get frame        │           │ Wait for target      │
│ 2. Detect face*     │           │ ↓                    │
│ 3. Set target ──────┼──────────>│ Send command         │
│ 4. Render overlay   │           │ ↓                    │
│ 5. Display          │           │ Wait for status      │
└─────────────────────┘           │ ↓                    │
   ~50ms/frame = 20 FPS            │ time.sleep(0.35s)    │
                                   └──────────────────────┘
```

*Face detection runs every 2nd frame (configurable)

### 2. Frame Skipping for Face Detection
- Detection every `detect_frame_skip=2` frames
- Cached results used between detections
- Reduces CPU load ~50% while maintaining tracking accuracy

## Changes Made

### ZoomCam Process (`zoomcam_process`)
1. **New motor control thread**:
   - Runs `motor_control_thread()` in background
   - Handles all serial I/O and settling delays
   - Main loop just updates `motor_target_deg[0]`

2. **Frame skip optimization**:
   - `detect_frame_counter` tracks when to run detector
   - `detect_frame_skip=2` (runs every other frame)
   - Cached face position used between detections

3. **Non-blocking architecture**:
   - Main loop never calls `time.sleep()`
   - Motor commands queued, not waited for
   - Frame processing decoupled from motor latency

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| ZoomCam FPS | 1-2 | 15-20 |
| Face detection rate | 1-2 Hz | 7-10 Hz |
| Motor latency | Blocks loop | Background |
| Tracking stability | Poor (oscillation) | Smooth |

## Tuning Parameters

```python
# In CONFIG section:
detect_frame_skip = 2  # Lower = more CPU, higher accuracy
                       # Higher = less CPU, slightly delayed

# Existing params still apply:
SEND_PERIOD = 0.25      # Min time between motor commands
SETTLING_DELAY = 0.35   # Motor settling (now in background)
MAX_STEP_WHEEL = 2      # Max degrees per command
```

## Testing
Run the tracker and observe:
- ZoomCam window should show **15-20 FPS** in top-left
- Face tracking should be smooth, not jerky
- Motor should move continuously without pausing the display
- Console should show motor commands happening in parallel

## Rollback
If issues occur, change:
```python
detect_frame_skip = 1  # Run detection every frame (original behavior)
```
