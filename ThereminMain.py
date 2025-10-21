"""
Theremin-style hand-tracking synthesizer.
- Left hand controls volume (0..1)
- Right hand controls pitch (0..1) mapped to 12-TET frequencies
"""

import time, math, cv2
import numpy as np
import HandTrackingModule as HandTracker
from AudioSynthesizer import AudioSynthesizer  # custom synth class

# =========================================================
# ---------------------- CONFIG ----------------------------
# =========================================================

# Camera
CAM_WIDTH, CAM_HEIGHT = 1280, 720
DETECTION_CONFIDENCE = 0.6
TOP_PADDING, BOTTOM_PADDING = 50, 50

# Pitch control (12-TET)
BASE_NOTE_FREQ = 261.63  # C4
OCTAVES = 1              # total octave span (e.g., C4 -> C6)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PREV_PITCH_MOD = 0.0
PITCH_MOD_SMOOTHING_FACTOR = 0.1 # smaller = smoother
PITCH_MOD_DEADZONE = 0.025

# Volume control
MIN_AMP = 0.0
MAX_AMP = 0.7
AMP_CURVE = 1.5  # gamma curve for perceptual volume control

# Drawing
FRAME_WAIT_MS = 1
BAR_WIDTH = 60
LEFT_BAR_X = 40
RIGHT_BAR_X = CAM_WIDTH - 100
BAR_COLOR_PITCH = (180, 0, 180)
BAR_COLOR_VOLUME = (0, 180, 0)
BAR_OUTLINE = (200, 200, 200)
TEXT_COLOR = (255, 255, 255)

# =========================================================
# ------------------- HELPER FUNCTIONS ---------------------
# =========================================================

def normalized_to_frequency_12TET(norm, base_freq=BASE_NOTE_FREQ, octaves=OCTAVES):
    """
    Convert a normalized 0..1 value to a quantized 12-TET frequency.
    """
    total_semitones = octaves * 12
    semitone_index = round(norm * total_semitones)
    freq = base_freq * 2 ** (semitone_index / 12)
    return freq

def normalized_to_frequency_microtonal(norm, base_freq=BASE_NOTE_FREQ, octaves=OCTAVES, divisions_per_octave=24):
    """
    Convert a normalized 0..1 value to a microtonal frequency scale.
    :param norm: normalized 0..1 value
    :param base_freq: base frequency (Hz)
    :param octaves: number of octaves to span
    :param divisions_per_octave: number of microtonal divisions per octave (e.g., 24 for quarter tones)
    """
    total_steps = octaves * divisions_per_octave
    step_index = round(np.clip(norm, 0.0, 1.0) * total_steps)
    freq = base_freq * 2 ** (step_index / divisions_per_octave)
    return freq


def normalized_to_frequency_continuous(norm, base_freq=BASE_NOTE_FREQ, octaves=OCTAVES):
    """
    Continuous logarithmic pitch mapping — no quantization.
    Ideal for pure theremin-like glides.
    """
    norm = np.clip(norm, 0.0, 1.0)
    return base_freq * (2 ** (octaves * norm))

def normalized_to_amplitude(norm_value, a_min=MIN_AMP, a_max=MAX_AMP, curve=AMP_CURVE):
    v = np.clip(norm_value, 0.0, 1.0) ** curve
    return a_min + (a_max - a_min) * v

def setup_camera(cam_index=0, width=CAM_WIDTH, height=CAM_HEIGHT):
    cap = cv2.VideoCapture(cam_index)
    cap.set(3, width)
    cap.set(4, height)
    return cap

# =========================================================
# --------------------- MAIN LOOP --------------------------
# =========================================================

def main():
    print("Starting a super cool Python Thermin...")

    # Init camera and hand tracker
    cap = setup_camera()
    detector = HandTracker.HandDetector(detectionCon=DETECTION_CONFIDENCE, maxHands=2)

    # Init synth
    synth = AudioSynthesizer(init_freq=BASE_NOTE_FREQ, init_amp=0.0)

    prev_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Camera read failed")
                break

            img = cv2.flip(frame, 1)
            img = detector.findHands(img)
            h, w, _ = img.shape

            volume_value = 0.0
            pitch_value = 0.0
            pitch_mod = 0.0
            angle_deg = 0.0
            
            # --- Hand tracking ---
            results = getattr(detector, "results", None)
            if results and results.multi_hand_landmarks and results.multi_handedness:
                for idx, handedness in enumerate(results.multi_handedness):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    lmList = detector.findPosition(img, handNum=idx, draw=False)
                    if not lmList:
                        continue

                    wrist_ids = [0, 1, 5, 9, 13, 17]
                    avg_x = sum(lmList[i][1] for i in wrist_ids) / len(wrist_ids)
                    avg_y = sum(lmList[i][2] for i in wrist_ids) / len(wrist_ids)
                    
                    #norm_y = 1.0 - (avg_y / h)
                    effective_height = h - (TOP_PADDING + BOTTOM_PADDING)
                    norm_y = 1.0 - ((avg_y - TOP_PADDING) / effective_height)
                    norm_y = float(np.clip(norm_y, 0.0, 1.0))

                    if label == "Left":
                        volume_value = float(np.clip(norm_y, 0.0, 1.0))
                    else:
                        pitch_value = float(np.clip(norm_y, 0.0, 1.0))
                        
                        # --- Pitch modulation calculation (index finger tilt) --- #
                        if len(lmList) > 8:  # make sure index finger landmarks exist
                            base = lmList[5]  # index base
                            tip = lmList[8]   # index tip

                            dx = tip[1] - base[1]
                            dy = tip[2] - base[2]
                            angle_deg = math.degrees(math.atan2(dy, dx))
                            angle_deg = (360 - angle_deg) % 360
                            
                            # Compute pitch modulation relative to 180°
                            raw_pitch_mod = (180 - angle_deg) / 45.0 # center at 180°, 45° = full deflection
                            raw_pitch_mod = float(np.clip(raw_pitch_mod, -1.0, 1.0))

                            # Exponential smoothing
                            global PREV_PITCH_MOD
                            pitch_mod = (PITCH_MOD_SMOOTHING_FACTOR * raw_pitch_mod) + ((1 - PITCH_MOD_SMOOTHING_FACTOR) * PREV_PITCH_MOD)
                            PREV_PITCH_MOD = pitch_mod
                            
                            # Deadzone
                            if abs(pitch_mod) < PITCH_MOD_DEADZONE:
                                pitch_mod = 0.0
                        else:
                            pitch_mod = 0.0

                    cv2.circle(img, (int(avg_x), int(avg_y)), 8, (0, 255, 255), cv2.FILLED)
                    cv2.putText(img, f"{label}", (int(avg_x) - 20, int(avg_y) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

            # --- Map to sound ---
            freq = normalized_to_frequency_continuous(pitch_value)
            amp = normalized_to_amplitude(volume_value)
            synth.set_frequency(freq)
            synth.set_amplitude(amp)
            synth.set_pitch_mod(pitch_mod)

            # --- Info text ---
            cv2.putText(img, f"Pitch: {pitch_value:.2f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            cv2.putText(img, f"Volume: {volume_value:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            cv2.putText(img, f"Freq: {int(freq)} Hz", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            cv2.putText(img, f"Pitch Mod: {pitch_mod:+.2f}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            # =========================================================
            # ---------------- DRAW BARS -------------------------------
            # =========================================================

            # Compute the usable region height (matches normalization)
            effective_height = h - (TOP_PADDING + BOTTOM_PADDING)

            # --- Volume bar (left) ---
            vol_h = int(volume_value * effective_height)
            cv2.rectangle(img, (LEFT_BAR_X, h - BOTTOM_PADDING - vol_h),
                        (LEFT_BAR_X + BAR_WIDTH, h - BOTTOM_PADDING), BAR_COLOR_VOLUME, cv2.FILLED)
            cv2.rectangle(img, (LEFT_BAR_X, TOP_PADDING),
                        (LEFT_BAR_X + BAR_WIDTH, h - BOTTOM_PADDING), BAR_OUTLINE, 2)

            # --- Pitch bar (right) with 12-TET note markers ---
            pitch_bar_x1 = RIGHT_BAR_X
            pitch_bar_x2 = RIGHT_BAR_X + BAR_WIDTH

            # Outline the bar region (matching active Y range)
            cv2.rectangle(img, (pitch_bar_x1, TOP_PADDING),
                        (pitch_bar_x2, h - BOTTOM_PADDING), BAR_OUTLINE, 2)

            # Fill current pitch
            pitch_h = int(pitch_value * effective_height)
            cv2.rectangle(img, (pitch_bar_x1, h - BOTTOM_PADDING - pitch_h),
                        (pitch_bar_x2, h - BOTTOM_PADDING), BAR_COLOR_PITCH, cv2.FILLED)

            # --- Draw note divisions + labels ---
            total_semitones = OCTAVES * 12
            for semitone_index in range(total_semitones + 1):
                semitone_norm = semitone_index / total_semitones
                # Map to visual Y range (respecting padding)
                y_pos = int(h - BOTTOM_PADDING - semitone_norm * effective_height)
                note_name = NOTE_NAMES[semitone_index % 12]
                octave_num = 4 + (semitone_index // 12)

                # tick line
                cv2.line(img, (pitch_bar_x1, y_pos), (pitch_bar_x1 + 10, y_pos), (255, 255, 255), 1)

                # label white notes only (to reduce clutter)
                if "#" not in note_name:
                    label = f"{note_name}{octave_num}"
                    cv2.putText(img, label, (pitch_bar_x2 + 5, y_pos + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)


            # --- FPS ---
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(img, f"FPS: {int(fps)}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show frame
            cv2.imshow("Theremin - Hand Tracking -> Synth", img)

            if cv2.waitKey(FRAME_WAIT_MS) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        print("Stopping synth and camera...")
        synth.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


# =========================================================
if __name__ == "__main__":
    main()
