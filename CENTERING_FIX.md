# Face Centering Fix - Logic Corrections

## Problems Identified

### 1. **Dead Zone Too Large** ❌
```python
ZOOMCAM_CENTER_DEAD_ZONE = 0.10  # 10% = ±36 pixels at 360p
```
- Face could be **36 pixels off-center** and system wouldn't correct
- That's nearly 1/5 of the frame height!

### 2. **Target Change Threshold Too High** ❌
```python
ZOOMCAM_TARGET_CHANGE_THRESHOLD = 0.7  # degrees
```
- Small corrections (< 0.7°) were ignored
- Prevented fine-tuning to exact position

### 3. **Integral Bias Reset Bug** ❌
**Old Logic**:
```python
if abs(center_error) > DEAD_ZONE:
    # Apply corrections + update integral
    adaptive_bias = fine_bias_deg
else:
    # Face centered - ZERO everything!
    fine_correction = 0.0
    adaptive_bias = 0.0  # ← BUG: Lost learned offset
```

**Problem**: When face entered dead zone, system forgot the learned integral bias that compensated for mapping errors. Next frame, it would move back out of dead zone, relearn, repeat → oscillation.

---

## Solutions Applied ✅

### 1. **Tighter Dead Zone**
```python
ZOOMCAM_CENTER_DEAD_ZONE = 0.04  # 4% = ±14 pixels at 360p
```
- Face must be within **±14 pixels** of center (Y = 0.46 to 0.54)
- Much more accurate centering

### 2. **Lower Target Threshold**
```python
ZOOMCAM_TARGET_CHANGE_THRESHOLD = 0.3  # degrees
```
- Now accepts corrections as small as 0.3°
- Allows fine positioning

### 3. **Fixed Integral Bias Logic** ✅
**New Logic**:
```python
if abs(center_error) > DEAD_ZONE:
    # Off-center: apply P correction + update integral
    fine_correction = calculate_proportional()
    fine_bias_deg += integrate_error()
else:
    # Centered: stop P correction but KEEP learned bias
    fine_correction = 0.0
    # fine_bias_deg unchanged ← Maintains learned offset

# ALWAYS apply integral bias (even when centered)
adaptive_bias = fine_bias_deg
target_wheel_deg += adaptive_bias
```

**Result**: 
- System learns mapping error via integral
- Maintains correction even when face is centered
- No oscillation because learned offset persists

---

## New Behavior

### When Face Detected:
1. **Approach**: Motor moves to mapped position from Arducam (e.g., -12°)
2. **Initial correction**: Face at Y=0.60, error=0.10 → applies +1.8° correction
3. **Learning**: Integral accumulates: 0.5° → 1.0° → 1.5° as face settles
4. **Settling**: Face reaches Y=0.48-0.52 (within dead zone)
5. **Stable**: Motor holds at `mapped_pos + learned_bias` → face stays centered

### Visual Feedback (ZoomCam window):
- `Error: 0.012` ← Shows actual Y deviation
- `Fine corr: +0.00°` ← P term (0 when in dead zone)
- `I-bias: +1.5°` ← Learned offset (persists)
- `✓ CENTERED` ← Green when error < 0.04
- `ADJUSTING (0.082)` ← Orange when correcting

### Console Output (improved):
```
[ZoomCam] Target -10.5°, move -2° → actual -10.0°
[ZoomCam] Target -11.0°, move -1° → actual -11.0°
[ZoomCam] Target -11.3°, move +0° → actual -11.0°  ← Within tolerance, stops
✓ CENTERED  ← Face stable at Y=0.49
```

---

## Key Insight

The **integral term** compensates for systematic errors:
- Mapping inaccuracies (100 samples still has gaps)
- Camera alignment offsets
- Mechanical play in wheel assembly

By **preserving** the learned bias when entering the dead zone, the system:
- Converges to true center (not just "close enough")
- Stops oscillating (no reset→relearn loop)
- Holds position accurately (bias compensates for mapping error)

---

## Tuning Guide

If still not centered well:

```python
# Stricter centering (face must be more precise)
ZOOMCAM_CENTER_DEAD_ZONE = 0.03  # ±11 pixels

# Allow smaller motor movements
ZOOMCAM_TARGET_CHANGE_THRESHOLD = 0.2

# Faster integral learning
ZOOMCAM_FINE_I_GAIN = 7.0  # (was 5.5)

# Tighter motor tolerance
WHEEL_TARGET_TOLERANCE = 0.5  # (was 1.0)
```

If overshooting/oscillating:

```python
# Slower integral learning
ZOOMCAM_FINE_I_GAIN = 3.0

# Smaller proportional gain
ZOOMCAM_FINE_KP = 12.0  # (was 18.0)
```
