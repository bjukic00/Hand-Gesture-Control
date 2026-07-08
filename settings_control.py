import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from tensorflow.keras.models import load_model

# --------------------------- Windows volume (pycaw) ---------------------------
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    VOLUME_CONTROL_AVAILABLE = True
except ImportError:
    print("Warning: Volume control not available. Install pycaw for Windows volume control.")
    VOLUME_CONTROL_AVAILABLE = False


# --------------------------- Helpers ---------------------------
def bbox_from_landmarks(lm_norm, pad=0.35):
    xs = [lm.x for lm in lm_norm]; ys = [lm.y for lm in lm_norm]
    if not xs or not ys: return None
    x1, y1 = max(0.0, min(xs) - pad), max(0.0, min(ys) - pad)
    x2, y2 = min(1.0, max(xs) + pad), min(1.0, max(ys) + pad)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: return None
    # make square
    if w > h:
        d = (w - h) / 2; y1 = max(0.0, y1 - d); y2 = min(1.0, y2 + d)
    else:
        d = (h - w) / 2; x1 = max(0.0, x1 - d); x2 = min(1.0, x2 + d)
    return [x1, y1, x2, y2]

def crop_letterbox_bgr(frame_bgr, bbox_norm, out_size=(64, 64)):
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_norm
    X1, Y1 = int(x1*W), int(y1*H)
    X2, Y2 = int(x2*W), int(y2*H)
    X1 = max(0, min(W-1, X1)); X2 = max(0, min(W,   X2))
    Y1 = max(0, min(H-1, Y1)); Y2 = max(0, min(H,   Y2))
    if X2 <= X1 or Y2 <= Y1: return None
    roi = frame_bgr[Y1:Y2, X1:X2]
    h, w = roi.shape[:2]; th, tw = out_size
    scale = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(roi, (nw, nh))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x_off, y_off = (tw-nw)//2, (th-nh)//2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas

def preprocess_for_cnn_from_bgr(bgr_img, img_size=None):
    if bgr_img is None or bgr_img.size == 0:
        return None
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Resize only if not already the right size
    if img_size is not None:
        h, w = rgb.shape[:2]
        if (h, w) != img_size:
            rgb = cv2.resize(rgb, (img_size[1], img_size[0]))  # (W, H)

    x = rgb.astype(np.float32) / 255.0  # <-- normalization [0,1]
    return x[None, ...]  

def setup_volume_control():
    if not VOLUME_CONTROL_AVAILABLE: return None
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as e:
        print(f"❌ Volume control initialization failed: {e}")
        return None

def set_volume_from_scalar(volume_interface, volume_percent):
    if volume_interface is None: return
    try:
        scalar = float(np.clip(volume_percent, 0, 100)) / 100.0
        volume_interface.SetMasterVolumeLevelScalar(scalar, None)
    except Exception as e:
        print(f"Error setting volume: {e}")


# --------------------------- Main ---------------------------
def main(model_path='best_gesture_model.keras', class_indices=None, img_size=(64, 64)):
    print("🚀 Starting Hand Gesture Controller...")
    model = load_model(model_path, compile=False)
    print(f"✓ CNN loaded. input={model.input_shape} output={model.output_shape}")

    # Label map
    if class_indices:
        idx_to_class = {v: k for k, v in class_indices.items()}
    else:
        idx_to_class = {0: 'Fist', 1: 'None', 2: 'Other', 3: 'Point', 4: 'Scale'}
    class_to_idx = {v:k for k,v in idx_to_class.items()}
    IDX_FIST  = class_to_idx['Fist']
    IDX_POINT = class_to_idx['Point']

    # MediaPipe (geometry only)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    MP_SMALL = (320, 240)

    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open camera"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Volume
    volume = setup_volume_control()
    try:
        smoothed_vol = float(volume.GetMasterVolumeLevelScalar()*100.0) if volume else 50.0
    except:
        smoothed_vol = 50.0

    # -------- Raw-prob thresholds (no EMA) --------
    ENTER_T = 0.6     # to activate a gesture
    EXIT_T  = 0.35     # to leave active gesture (generic)
    DWELL_N = 2        # frames required to enter
    EXIT_N = 2         # frames required to exit
    POINT_EXIT_T = 0.3  # quicker release for Point
    POINT_MIN_HOLD = 5
    point_hold_frames = 0
    exit_count = 0
    dwell_count = 0
    
    # Fist click (single edge-trigger while pointing)
    FIST_PRESS_T_POINT   = 0.50   # press threshold (raw)
    FIST_RELEASE_T_POINT = 0.20   # release threshold (raw)
    fist_high = False             # state for edge detection
    CLICK_COOLDOWN = 12           # ~0.2s @ 30fps
    click_cooldown = 0

    # Cursor (palm center)
    screen_w, screen_h = pyautogui.size()
    alpha_cursor = 0.9 # fraction of a gap you move toward new raw target
    deadzone = 2
    x_s, y_s = screen_w // 2, screen_h // 2
    prev_x, prev_y = x_s, y_s

    # Volume calibration + mapping (kept)
    ALPHA_VOL = 0.20
    calib_min = calib_max = None
    RANGE_SHRINK = 0.8
    EDGE_SNAP = 3.0

    # BBox smoothing
    bbox_prev = None
    ALPHA_BBOX = 0.40

    # Palm center landmarks - for cursor movement
    PALM_IDXS = [
        mp_hands.HandLandmark.WRIST,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    active_gesture = 'None'
    last_active_gesture = 'None'
    last_label = None

    print("\n🎮 Point=mouse (PALM CENTER) • Fist=click • Scale=volume • q=quit")

    try:
        while True:
            ret, frame_bgr = cap.read() # ret is success flag
            if not ret: break
            frame_bgr = cv2.flip(frame_bgr, 1) # horizontal flip
            frame = frame_bgr 

            # ---------- MediaPipe (downscaled) ----------
            rgb_small = cv2.cvtColor(cv2.resize(frame_bgr, MP_SMALL), cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_small)
            have_hand = results.multi_hand_landmarks is not None
            lm_norm = results.multi_hand_landmarks[0].landmark if have_hand else None # take landmarks of the first hand

            # Build (smoothed) square bbox
            if lm_norm:
                bbox_now = bbox_from_landmarks(lm_norm, pad=0.35)
                if bbox_now and bbox_prev:
                    bbox_prev = [ALPHA_BBOX*bbox_prev[i] + (1-ALPHA_BBOX)*bbox_now[i] for i in range(4)] 
                else:
                    bbox_prev = bbox_now

            if bbox_prev is None:
                cv2.putText(frame, "Detecting hand...", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                cv2.imshow('Hand Gesture Control', frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

            roi_bgr = crop_letterbox_bgr(frame_bgr, bbox_prev, out_size=img_size)
            if roi_bgr is None:
                cv2.imshow('Hand Gesture Control', frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

            # ---------- CNN classification (RAW softmax) ----------
            x = preprocess_for_cnn_from_bgr(roi_bgr, img_size)
            p = model.predict(x, verbose=0)[0]   # raw probs
            pred_idx = int(np.argmax(p))
            pred_lbl = idx_to_class.get(pred_idx, 'None')
            conf     = float(p[pred_idx])

            # Hysteresis + dwell (using RAW probs)
            if pred_lbl == last_label: dwell_count += 1
            else:
                dwell_count = 1
                last_label = pred_lbl

            if active_gesture == 'None':
                if pred_lbl in ('Point','Scale') and conf >= ENTER_T and dwell_count >= DWELL_N:
                    active_gesture = pred_lbl
                    if pred_lbl == 'Point':
                        point_hold_frames = POINT_MIN_HOLD
            else:
                # release logic
                if active_gesture == 'Point':
                    if p[IDX_POINT] < POINT_EXIT_T:
                        exit_count += 1
                    else:
                        exit_count = 0
                else:
                    if p[class_to_idx[active_gesture]] < EXIT_T:
                        exit_count += 1
                    else:
                        exit_count = 0

                if active_gesture == 'Point' and point_hold_frames > 0:
                    point_hold_frames -= 1
                    exit_count = 0
                elif exit_count >= EXIT_N:
                    active_gesture = 'None'
                    exit_count = 0

            # ---------- Single-click edge trigger while pointing ----------
            if active_gesture == 'Point':
                if not fist_high and p[IDX_FIST] >= FIST_PRESS_T_POINT and click_cooldown == 0:
                    # rising edge => click once
                    pyautogui.click()
                    click_cooldown = CLICK_COOLDOWN
                    cv2.putText(frame, 'CLICK!', (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                    fist_high = True
                elif fist_high and p[IDX_FIST] <= FIST_RELEASE_T_POINT:
                    # re-arm after dropping below release threshold
                    fist_high = False
            else:
                # if not pointing, just re-arm based on release
                if p[IDX_FIST] <= FIST_RELEASE_T_POINT:
                    fist_high = False

            # HUD
            top2 = np.argsort(p)[-2:][::-1] # takes higest probabilites and reverses them from top to down
            txt = f"{idx_to_class[top2[0]]}:{p[top2[0]]:.2f} | {idx_to_class[top2[1]]}:{p[top2[1]]:.2f}"
            cv2.putText(frame, f'Active: {active_gesture}', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            cv2.putText(frame, f'Conf:{conf:.2f}  {txt}', (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            entered_scale = (active_gesture == 'Scale' and last_active_gesture != 'Scale')

            # ---------- Actions ----------
            if have_hand:
                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS) # hand skeleton

            if active_gesture == 'Point' and have_hand:
                # Compute the palm center as the average of some landmark coordinates
                cx = np.mean([lm_norm[i].x for i in PALM_IDXS])
                cy = np.mean([lm_norm[i].y for i in PALM_IDXS])
                # Convert normalized hand coordinates into pixel coordinates of the screen.
                x_raw = int(cx * screen_w)
                y_raw = int(cy * screen_h)
                # Apply exponential smoothing with factor alpha_cursor
                x_s = int((1 - alpha_cursor) * x_s + alpha_cursor * x_raw)
                y_s = int((1 - alpha_cursor) * y_s + alpha_cursor * y_raw)
                # Only move the mouse if the new smoothed position differs enough from the previous position.
                if abs(x_s - prev_x) + abs(y_s - prev_y) > deadzone:
                    pyautogui.moveTo(x_s, y_s, duration=0)
                    prev_x, prev_y = x_s, y_s

            elif active_gesture == 'Scale' and have_hand and VOLUME_CONTROL_AVAILABLE and volume is not None:
                H, W = frame.shape[:2]
                thumb_tip = lm_norm[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = lm_norm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = lm_norm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                pinky_mcp = lm_norm[mp_hands.HandLandmark.PINKY_MCP]
                # Calculates the Euclidean distance (in pixels) between thumb tip and index tip
                pinch_px = np.hypot((thumb_tip.x - index_tip.x) * W, (thumb_tip.y - index_tip.y) * H)
                # Measures the width of your hand in pixels
                hand_w_px = max(1.0, np.hypot((index_mcp.x - pinky_mcp.x) * W, (index_mcp.y - pinky_mcp.y) * H))
                norm = pinch_px / hand_w_px

                # dynamic calibration
                # creates a starting range based on your hand’s first position.
                if calib_min is None or calib_max is None:
                    calib_min, calib_max = norm * 0.6, norm * 1.4
                else:
                    calib_min = min(calib_min *  0.98 + norm * 0.02, norm)
                    calib_max = max(calib_max *  0.98 + norm * 0.02, norm)
                    if calib_max - calib_min < 0.10:
                        calib_min -= 0.05; calib_max += 0.05

                # shrink mapping range so ends are easier
                # find the current calibration window center & half-width
                mid  = 0.5 * (calib_min + calib_max)
                half = 0.5 * (calib_max - calib_min) * RANGE_SHRINK
                map_min, map_max = mid - half, mid + half
                norm_c = float(np.clip(norm, map_min, map_max))
                target = np.interp(norm_c, [map_min, map_max], [0, 100])

                # snap near edges + smoothing
                if target >= 100 - EDGE_SNAP: target = 100.0
                elif target <= EDGE_SNAP:     target = 0.0

                if entered_scale: smoothed_vol = target
                else:
                    alpha = ALPHA_VOL if (EDGE_SNAP < target < 100-EDGE_SNAP) else min(0.75, ALPHA_VOL + 0.35)
                    smoothed_vol = (1 - alpha) * smoothed_vol + alpha * target
                
                #vol_to_set = int(np.ceil(smoothed_vol)) 
                vol_to_set = int(round(smoothed_vol / 2.0) * 2)
                set_volume_from_scalar(volume, vol_to_set)

            # cooldown
            if click_cooldown > 0: click_cooldown -= 1

            # Volume HUD
            if VOLUME_CONTROL_AVAILABLE and volume:
                try:
                    now_vol = int(volume.GetMasterVolumeLevelScalar() * 100)
                    cv2.putText(frame, f'Vol: {now_vol}%', (frame.shape[1] - 120, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                except: pass

            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow('Hand Gesture Control', frame)

            last_active_gesture = active_gesture
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        cap.release(); cv2.destroyAllWindows(); hands.close(); print("\n✓ Cleanup completed")


if __name__ == "__main__":
    MODEL_PATH = 'hand_gesture_cnn_dp.keras'
    CLASS_INDICES = {'Fist': 0, 'None': 1, 'Other': 2, 'Point': 3, 'Scale': 4}
    main(model_path=MODEL_PATH, class_indices=CLASS_INDICES, img_size=(64, 64))
