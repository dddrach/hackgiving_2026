"""Turkey keyboard: DOWN/T/UP select position; O switches mode; X deletes; Y accepts.

Test gestures and pad keys without hardware:
    python turkey_keyboard.py --simulate --no-camera --pads 10
Test actual webcam with simulated foot pads:
    python turkey_keyboard.py --simulate --pads 10
Run with Arduino connected:
    python turkey_keyboard.py --port COM4 --pads 10
Type into a focused application (optional):
    python turkey_keyboard.py --port COM4 --pads 10 --type-to-app

Safety: preview-only by default. Never test OS-typing against sensitive apps.
"""
import argparse
import time
import urllib.request
from pathlib import Path
from prediction import WordPredictor
from gesture_calibration import GestureCalibration, arm_features

from keyboard_logic import (GestureHold, PadPresses, GESTURE_SLOTS, GESTURE_HOLDS, GESTURE_ACTIONS, mapped_key,
                            apply_key, classify_gesture)
from keyboard_logic import accept_tracking_ready, arms_lowered

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "pose_landmarker_lite.task"
OUTPUT_PATH = HERE / "typed_output.txt"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
SIM_KEYS = "1234567890-="  # 1..9, 0 means pad 10, - means 11, = means 12
WINDOW_NAME = "Turkey Keyboard - q to quit"


def screen_work_area():
    """Usable screen size (excluding the taskbar), or None if unknown."""
    try:
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(rect), 0):
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        pass
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (0, 0, root.winfo_screenwidth(), root.winfo_screenheight() - 60)
        root.destroy()
        return size
    except Exception:
        return None


def fit_window_to_screen(cv2, width, height):
    """Shrink the preview window so the whole UI (pads + typed text) fits on screen."""
    area = screen_work_area()
    if not area:
        return
    left, top, sw, sh = area
    title_bar = 40
    scale = min(1.0, (sw * 0.98) / width, (sh - title_bar) / height)
    w, h = int(width * scale), int(height * scale)
    cv2.resizeWindow(WINDOW_NAME, w, h)
    cv2.moveWindow(WINDOW_NAME, left + max(0, (sw - w) // 2), top)


def load_model():
    """Only called if camera mode is enabled; downloads official model once."""
    import mediapipe as mp
    if not MODEL_PATH.is_file():
        print("Downloading MediaPipe pose model (~6MB) on first run...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as exc:
            raise SystemExit(
                f"Could not download pose model: {exc}\n"
                f"Download {MODEL_URL}\n"
                f"and save it as {MODEL_PATH}") from exc
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, vision.PoseLandmarker.create_from_options(options)


def open_serial(port):
    import serial
    try:
        return serial.Serial(port, 115200, timeout=0)
    except serial.SerialException as exc:
        raise SystemExit(f"Cannot open Arduino port {port}: {exc}\n"
                         "Check the correct port; close Arduino Serial Monitor.") from exc


def draw_skeleton(cv2, frame, landmarks):
    if not landmarks:
        return
    h, w = frame.shape[:2]
    segments = ((11,12), (11,13), (13,15), (12,14), (14,16),
                (11,23), (12,24), (23,24))
    for a, b in segments:
        pa, pb = landmarks[a], landmarks[b]
        if pa.visibility is not None and min(pa.visibility, pb.visibility) < 0.5:
            continue
        cv2.line(frame, (round(pa.x*w), round(pa.y*h)),
                 (round(pb.x*w), round(pb.y*h)), (104, 234, 135), 3)
    for i in (0,11,12,13,14,15,16):
        p = landmarks[i]
        cv2.circle(frame, (round(p.x*w), round(p.y*h)), 5, (60,224,255), -1)


def render_ui(cv2, image, args, mode, slot, raw, latch, pads, typed, status, simulator,
              suggestion='', calibration_hint=''):
    """Draw current state and pad mappings below camera image."""
    h, w = image.shape[:2]
    view = cv2.copyMakeBorder(image, 118, 246, 0, 0,
                              cv2.BORDER_CONSTANT, value=(27, 31, 39))
    white = (240, 245, 249)
    teal = (140, 225, 120)
    muted = (185, 188, 194)
    orange = (65, 190, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    slot_label = "NOT SELECTED" if slot is None else ("FIRST", "SECOND", "THIRD")[slot]
    if args.pads == 9 and mode == 'NUMBERS' and slot is not None:
        slot_label = ('1-9', '0 / EDIT', 'UNASSIGNED')[slot]
    cv2.putText(view, f"TURKEY KEYBOARD  |  {mode}  |  {slot_label}",
                (13, 34), font, 0.84, white, 2, cv2.LINE_AA)
    instructions = 'DOWN=1st   T=2nd   UP=3rd   O=switch letters/numbers'
    if args.pads == 9 and mode == 'NUMBERS':
        instructions = 'DOWN=1-9   T=0 / Space / Delete / Enter   O=letters'
    cv2.putText(view, instructions,
                (13, 62), font, 0.57, teal, 2, cv2.LINE_AA)
    cv2.putText(view, f"Camera sees: {raw or '-'}    Locked: {slot_label}"
                + ("    [OVERRIDE: press R]" if simulator else ""),
                (13, 88), font, 0.57, muted, 1, cv2.LINE_AA)
    sim_hint = '  |  r=clear test pose; 1-9 to step' if args.pads == 9 else '  |  r=clear test pose; 1-9/0/-/= to step'
    cv2.putText(view, "Q=quit" + (sim_hint if args.simulate else ""),
                (13, 109), font, 0.49, muted, 1, cv2.LINE_AA)
    if latch.candidate:
        duration = GESTURE_HOLDS.get(latch.candidate,.40)
        observed_until = latch.missing_since if latch.missing_since is not None else time.monotonic()
        progress = 1 if latch.fired else min(1, max(0, (observed_until-latch.since)/duration))
        cv2.rectangle(view, (w-190, 98), (w-14,110), (70,70,70), -1)
        cv2.rectangle(view, (w-190, 98), (w-190+round(176*progress),110), teal, -1)
    cv2.putText(view, status[:110], (13, 143+h), font, 0.63, orange, 2, cv2.LINE_AA)

    # Draw 8-12 pad cards with current active mapping; 6 per row.
    ncol = min(6, args.pads)
    gap, xmargin = 8, 14
    cell = (w - 2*xmargin - gap*(ncol-1)) // ncol
    for pad in range(args.pads):
        row, col = divmod(pad, ncol)
        x = xmargin + col*(cell + gap)
        y = h + 160 + row*52
        active = pad in pads.down_at
        dedicated = args.pads == 12 and pad >= 10
        cv2.rectangle(view, (x,y), (x+cell,y+45),
                      (65, 102, 74) if active else ((78, 65, 43) if dedicated else (53, 62, 77)), -1)
        cv2.rectangle(view, (x,y), (x+cell,y+45),
                      (115, 210, 134) if active else (92, 102, 117), 1)
        key = mapped_key(mode, slot, pad, args.pads) or "-"
        title = f"PAD {pad+1}"
        if dedicated:
            title += ' ALWAYS'
        elif mode == 'LETTERS':
            group = [mapped_key(mode,s,pad,args.pads) for s in range(3)]
            title += ' '+''.join(k if len(k)==1 else '_' for k in group)
        cv2.putText(view, title, (x+6,y+16), font, .38, muted, 1, cv2.LINE_AA)
        label = 'DELETE' if key == 'BACKSPACE' else key
        cv2.putText(view, label[:11], (x+6,y+36), font, .60, white, 2, cv2.LINE_AA)
    short_text = typed.replace("\n", " / ").replace("\t", " ")[-72:]
    # Keep the untyped suffix directly beside committed text and within the view.
    while short_text and cv2.getTextSize('TYPED: '+short_text+suggestion,
                                         font, .56, 1)[0][0] > w-28:
        short_text = short_text[1:]
    committed = 'TYPED: '+short_text
    cv2.putText(view, committed, (13, h+300), font, .56, white, 1, cv2.LINE_AA)
    offset = cv2.getTextSize(committed, font, .56, 1)[0][0]
    cv2.putText(view, suggestion, (13+offset, h+300), font, .56,
                (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(view, 'Y = ACCEPT suggestion | X = DELETE character | Tab also accepts',
                (13,h+325), font, .43, muted, 1, cv2.LINE_AA)
    action_hint = 'Accept: hold 0.65s | Delete: hold 1s | Lower arms before repeating.'
    if latch.action_locked:
        action_hint = 'ACTION DONE - lower your arms to rearm Accept / Delete.'
    if calibration_hint and not latch.action_locked:
        action_hint = calibration_hint
    if raw == 'Y' and not latch.action_locked:
        action_hint = 'ACCEPT detected - hold steady.' if suggestion else 'No suggestion to accept - type more letters.'
    cv2.putText(view, action_hint, (13,h+347), font, .43, teal, 1, cv2.LINE_AA)
    return view


def main():
    ap = argparse.ArgumentParser(description="Turkey hands-free CV + LDR keyboard")
    ap.add_argument("--pads", type=int, choices=range(8,13), default=9,
                    metavar="8-12", help="Number of installed LDR pads (default 9)")
    ap.add_argument("--port", help="Arduino serial port, e.g. COM4, /dev/ttyACM0")
    ap.add_argument("--simulate", action="store_true",
                    help="Simulate foot pad inputs with 1-9/0/-/= in the video window")
    ap.add_argument("--no-camera", action="store_true",
                    help='With --simulate: d/t/u select, o=mode, x=delete, y=accept; r=release')
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--type-to-app", action="store_true",
                    help="Send typed keys to currently focused application (CAREFUL)")
    args = ap.parse_args()

    if not args.simulate and not args.port:
        ap.error("Provide --port COM4 for Arduino, or --simulate to test without it")
    if args.no_camera and not args.simulate:
        ap.error("--no-camera must be used with --simulate")
    if args.pads == 9:
        print('9 pads: DOWN/T/UP select A-Z and Space (UP + pad 9).')
        print('Numbers: DOWN = 1-9; T = pad 1:0, pad 2:Space, pad 3:Delete, pad 4:Enter.')
    elif args.pads == 8:
        print('8 pads: Y/Z are unavailable; select O in numbers for remaining digits and edits.')

    # Keep all GUI and computer-vision imports out of tests.
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Missing OpenCV. Run: python -m pip install -r requirements.txt") from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Missing numpy dependency. Run: python -m pip install -r requirements.txt") from exc

    keyboard = None
    if args.type_to_app:
        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            keyboard = pyautogui
        except Exception as exc:
            raise SystemExit(f"PyAutoGUI could not initialize: {exc}\n"
                             "Try without --type-to-app to test the preview first.") from exc

    ser = None if args.simulate else open_serial(args.port)
    cam = None
    detector = None
    mp = None
    if not args.no_camera:
        cam = cv2.VideoCapture(args.camera)
        if not cam.isOpened():
            raise SystemExit("Cannot access webcam. Check camera permissions or --camera 1")
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        try:
            mp, detector = load_model()
        except ImportError as exc:
            raise SystemExit("MediaPipe not installed. Run: python -m pip install -r requirements.txt") from exc

    hold = GestureHold()
    calibration = GestureCalibration(HERE / 'gesture_calibration_positions.json',
                                     legacy_path=HERE / 'gesture_calibration_straight_up.json')
    predictor = WordPredictor(HERE / 'autocomplete_history.json')
    print(f'Offline autocomplete: {predictor.vocabulary_size} words. Tab accepts one word.')
    suggestion = ''
    suggestion_text = ''
    pads = PadPresses(args.pads, dwell=0.12)
    mode = "LETTERS"
    slot = None  # Require intentional initial pose; no surprising first key.
    typed = ""
    status = 'Select position: diagonal DOWN=first, T=second, straight UP=third.'
    rx_buffer = bytearray()
    simulated_presses = {}  # pad -> release timestamp
    pose_override = None
    accept_target = None
    previous_time_ms = -1
    t0 = time.monotonic()
    print('DOWN/T/UP: positions 1/2/3 (0.40s). O: switch mode (0.50s).')
    print('Y: accept suggestion (0.65s). X: delete one character (1 second).')
    print('Lower arms for 0.25s before repeating Accept, Delete or mode switch.')
    print('O: rounded, bent elbows. UP: straight elbows, hands together. C calibrates.')
    if args.simulate:
        print("Test foot pad: press 1..9, 0 (pad10), - (pad11), = (pad12) in GUI.")
        print('Pose test keys: d=DOWN, t=T, u=UP, o=O, y=Y, x=X; r=release.')
        if args.pads == 12:
            print('12-pad layout: minus (-) = Space; equals (=) = Delete, in every mode.')
    if keyboard:
        print("WARNING: OS typing enabled. Open and focus a blank text editor to receive keys!")
        print("Keys go to the currently focused app, NOT automatically to a document.")
    else:
        print("Safe preview only; output saved to typed_output.txt.")

    def type_selected(pad):
        nonlocal typed, status
        key = mapped_key(mode, slot, pad, args.pads)
        if key is None:
            status = f'Pad {pad+1} not assigned. Select DOWN / T / UP to choose a position.'
            return
        typed = apply_key(typed, key)
        if key not in ('BACKSPACE',):
            predictor.learn(typed)
        OUTPUT_PATH.write_text(typed, encoding="utf-8")
        if keyboard:
            try:
                if key in ("SPACE", "BACKSPACE", "ENTER"):
                    keyboard.press({"SPACE":"space", "BACKSPACE":"backspace",
                                    "ENTER":"enter"}[key])
                else:
                    keyboard.write(key, interval=0)
            except Exception as exc:
                status = f"Text saved; OS typing failed: {exc}"
                print(status)
                return
        status = f"TYPED {key}: pad {pad+1}, {mode}, " + (
            "number" if mode == "NUMBERS" else f"position {(slot or 0)+1}")
        print(status)

    def accept_suggestion():
        nonlocal typed, status, suggestion
        if not suggestion:
            status = 'No suggestion to accept. Keep typing.'
            return
        suffix = suggestion
        typed += suffix
        suggestion = ''
        OUTPUT_PATH.write_text(typed, encoding='utf-8')
        status = 'ACCEPTED WORD: '+typed.split()[-1]
        if keyboard:
            try:
                keyboard.write(suffix, interval=0)
            except Exception as exc:
                status = f'Text saved; OS typing failed: {exc}'
        print(status)

    def delete_character():
        nonlocal typed, status, suggestion
        if not typed:
            status = 'Nothing to delete.'
            return
        typed = apply_key(typed, 'BACKSPACE')
        suggestion = ''
        OUTPUT_PATH.write_text(typed, encoding='utf-8')
        status = 'DELETED ONE CHARACTER - lower arms before repeating.'
        if keyboard:
            try:
                keyboard.press('backspace')
            except Exception as exc:
                status = f'Text saved; OS typing failed: {exc}'
        print(status)

    # Resizable window (drag the corner or maximize) that keeps the aspect ratio.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    window_fitted = False

    try:
        if detector is not None:
            detector.__enter__()
        while True:
            now = time.monotonic()
            features = None
            if args.no_camera:
                frame = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(frame, "POSE SIMULATION (no webcam)", (115,235),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.10, (220,220,220), 2)
                cv2.putText(frame, 'd/t/u = positions 1/2/3; o=mode; y=accept; x=delete; r=release',
                            (100,300), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (170,205,185), 2)
                raw = pose_override
                landmarks = None
            else:
                ok, frame = cam.read()
                if not ok:
                    status = "Lost webcam video; check camera connection"
                    print(status)
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(previous_time_ms + 1, int((now-t0)*1000))
                previous_time_ms = timestamp_ms
                result = detector.detect_for_video(mp_frame, timestamp_ms)
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                raw = classify_gesture(landmarks, w, h)
                features = arm_features(landmarks,w,h)
                calibration.observe(features,now)
                raw = calibration.classify(features,raw)
                if raw in ('UP','Y') and not accept_tracking_ready(landmarks):
                    raw = None
                if pose_override is not None:
                    raw = pose_override
                draw_skeleton(cv2, frame, landmarks)
            if calibration.active:
                raw = None

            # Read the Arduino events emitted by any of the included sketches.
            if ser is not None:
                try:
                    rx_buffer.extend(ser.read(min(ser.in_waiting, 256)))
                except OSError as exc:
                    status = f"Arduino disconnected: {exc}"
                    print(status)
                    break
                while b"\n" in rx_buffer:
                    line, _, tail = rx_buffer.partition(b"\n")
                    rx_buffer = bytearray(tail)
                    parts = line.decode("utf-8", errors="replace").strip().split()
                    if len(parts) == 2 and parts[0] in ("DOWN", "UP"):
                        try:
                            pad = int(parts[1]) - 1
                            pads.on_event(parts[0], pad, now)
                        except ValueError:
                            continue
                    elif parts and parts[0] in ("READY", "BASELINE", "CALIBRATING:", "ERROR"):
                        print("ARDUINO:", " ".join(parts))
                if len(rx_buffer) > 1024:
                    rx_buffer.clear()

            for pad, release_at in list(simulated_presses.items()):
                if now >= release_at:
                    pads.on_event("UP", pad, now)
                    del simulated_presses[pad]

            # Gesture input does NOT have to coincide with a foot press.
            # The selection is remembered until the next intentional pose.
            if raw == 'Y' and hold.candidate != 'Y':
                accept_target = (typed,suggestion) if suggestion else None
            released = raw is None if args.no_camera or pose_override is not None else arms_lowered(landmarks)
            confirmed = hold.update(raw, now, allowed=(not pads.down_at and not calibration.active),
                                    release_observed=released)
            action = GESTURE_ACTIONS.get(confirmed)
            if action == 'MODE':
                mode = "NUMBERS" if mode == "LETTERS" else "LETTERS"
                status = f'MODE SWITCHED TO {mode}; lower arms before switching again.'
                print(status)
            elif confirmed in GESTURE_SLOTS:
                slot = GESTURE_SLOTS[confirmed]
                status = f"LOCKED POSITION {slot+1} ({confirmed} pose); you can lower arms now."
                print(status)
            elif action == 'ACCEPT':
                if suggestion and accept_target == (typed,suggestion):
                    accept_suggestion()
                else:
                    status = 'No unchanged suggestion to accept. Lower arms and try again.'
            elif action == 'DELETE':
                delete_character()

            ready_pad = pads.ready(now)
            if ready_pad is not None:
                if pads.blocked_until_clear:
                    status = "Multiple sensors activated: release all and retry."
                elif not calibration.active:
                    type_selected(ready_pad)

            new_suggestion = predictor.suggest(typed) if mode == 'LETTERS' else ''
            if suggestion_text != typed or new_suggestion != suggestion:
                suggestion_text = typed
                suggestion = new_suggestion

            view = render_ui(cv2, frame, args, mode, slot, raw, hold, pads,
                             typed, status, pose_override is not None,
                             suggestion, calibration.hint if calibration.active else
                             'Accept: 0.65s | Delete: 1s | C = calibrate gestures')
            cv2.imshow(WINDOW_NAME, view)
            if not window_fitted:
                fit_window_to_screen(cv2, view.shape[1], view.shape[0])
                window_fitted = True
            k = cv2.waitKey(1) & 0xff
            if k in (27, ord('q')):
                break
            if k == ord('c'):
                if args.no_camera:
                    status = 'Calibration needs the webcam. Use A/D to test simulated actions.'
                else:
                    calibration.capture(now)
                    pose_override = None
                    hold = GestureHold()
                    status = calibration.message or calibration.hint
            if k == 9 and not calibration.active:
                accept_suggestion()
            if args.simulate and k != 255:
                key = chr(k).lower()
                if key in ("t", "o", "y", "x"):
                    pose_override = key.upper()
                elif key in ('u','d','a'):
                    pose_override = {'u':'UP','d':'DOWN','a':'Y'}[key]
                elif key == "r":
                    pose_override = None
                elif key in SIM_KEYS:
                    p = SIM_KEYS.index(key)
                    if p < args.pads and p not in simulated_presses:
                        pads.on_event("DOWN", p, now)
                        simulated_presses[p] = now + 0.38

    except KeyboardInterrupt:
        pass
    finally:
        if detector:
            detector.__exit__(None, None, None)
        if cam is not None:
            cam.release()
        if ser is not None:
            ser.close()
        cv2.destroyAllWindows()
        print(f"Final text saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
