import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
VIDEO_PATH = "DJI_0829.MP4"
MODEL_PATH = "yolo11hbb_base90+40_poc_w_full_poc_blur_gripper.pt"
SEARCH_RADIUS = 250
NUM_FEATURES = 10
INIT_FRAMES = 5

def main():
    # Load model
    print("Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_PATH}")
        return

    # Tracking state
    features_prev = None  # Coordinates of features in previous frame
    features_init = None  # Coordinates of features at initialization (frame 4)
    center_init = None    # Center of object at initialization (frame 4)
    gray_prev = None      # Previous frame grayscale
    
    tracking_active = False
    frame_count = 0

    # Create resizable window
    cv2.namedWindow('Tracking System', cv2.WINDOW_NORMAL)
    
    # Calculate target display size for 4K side-by-side (32:9 aspect ratio)
    # Original: 3840x2160 -> Combined: 7680x2160
    # Target Width: 1900 (to fit on standard screen)
    # Target Height: 1900 * (2160 / 7680) = ~534
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_width = 1900
    target_height = int(target_width * (orig_h / (orig_w * 2)))
    
    cv2.resizeWindow('Tracking System', target_width, target_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create side-by-side canvas
        frame_left = frame.copy()
        frame_right = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detected_center = None
        status_text = "UNKNOWN"
        status_color = (100, 100, 100)

        # --- Phase 1: Initialization ---
        if frame_count < INIT_FRAMES:
            status_text = "INITIALIZING (AI)"
            status_color = (0, 255, 255) # Yellow
            
            # Run YOLO
            results = model(frame, verbose=False)
            best_conf = 0
            box = None
            
            for r in results:
                boxes = r.boxes
                for b in boxes:
                    conf = float(b.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        box = b.xyxy[0].cpu().numpy()

            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                detected_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                
                # Draw YOLO on Left
                cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.circle(frame_left, (int(detected_center[0]), int(detected_center[1])), 10, (0, 0, 255), -1)
                
                # Initialize Features
                if frame_count == INIT_FRAMES - 1:
                    print("Initializing tracking features...")
                    mask = np.zeros_like(gray)
                    cv2.circle(mask, (int(detected_center[0]), int(detected_center[1])), SEARCH_RADIUS, 255, -1)
                    
                    p0 = cv2.goodFeaturesToTrack(gray, maxCorners=NUM_FEATURES, qualityLevel=0.3, minDistance=10, mask=mask)
                    
                    if p0 is not None:
                        features_init = p0.reshape(-1, 2)
                        features_prev = features_init
                        center_init = detected_center
                        tracking_active = True
                        print(f"Initialized {len(features_init)} features.")

        # --- Phase 2: Tracking ---
        elif tracking_active:
            status_text = "TRACKING (OPTICAL FLOW)"
            status_color = (0, 255, 0) # Green
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, features_prev, None, winSize=(21, 21), maxLevel=3)
            
            if p1 is not None:
                st = st.flatten()
                valid_indices = (st == 1)
                current_points = p1[valid_indices]
                
                if len(current_points) >= 2:
                    features_init_subset = features_init[valid_indices]
                    M, inliers = cv2.estimateAffinePartial2D(features_init_subset, current_points)
                    
                    estimated_center = None
                    if M is not None:
                        c_x, c_y = center_init
                        est_x = M[0, 0] * c_x + M[0, 1] * c_y + M[0, 2]
                        est_y = M[1, 0] * c_x + M[1, 1] * c_y + M[1, 2]
                        estimated_center = (int(est_x), int(est_y))
                        
                        # VISUALIZATION
                        # Draw tracked points and lines
                        for i, (new, old) in enumerate(zip(current_points, features_init_subset)):
                            a, b = new.ravel()
                            cv2.circle(frame_right, (int(a), int(b)), 8, (255, 0, 0), -1) # Blue dots
                            if estimated_center:
                                cv2.line(frame_right, (int(a), int(b)), estimated_center, (0, 255, 255), 2)

                        if estimated_center:
                            cv2.drawMarker(frame_right, estimated_center, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=4)

                    features_prev = current_points.reshape(-1, 2)
                    features_init = features_init_subset.reshape(-1, 2)
                else:
                    status_text = "LOST TRACKING"
                    status_color = (0, 0, 255)
                    tracking_active = False
            else:
                status_text = "LOST ALL POINTS"
                status_color = (0, 0, 255)
                tracking_active = False

        # --- UI Overlay Helper ---
        def draw_ui(img, title, stat_txt, color):
            # Overlay box
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            # Title
            cv2.putText(img, title, (40, 90), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3)
            
            # Status
            (text_w, text_h), _ = cv2.getTextSize(stat_txt, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)
            cv2.putText(img, stat_txt, (img.shape[1] - text_w - 40, 90), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)
            
            # Info Stats
            info = f"Frame: {frame_count}"
            if tracking_active:
                info += f" | Features: {len(features_prev)}"
            cv2.putText(img, info, (40, 190), cv2.FONT_HERSHEY_DUPLEX, 1.2, (220, 220, 220), 2)

        draw_ui(frame_left, "YOLO Reference", "AI DETECTION", (0, 255, 0))
        draw_ui(frame_right, "Geometric Tracker", status_text, status_color)

        # Combine
        combined = np.hstack((frame_left, frame_right))
        
        # Draw Separator Line
        separator_x = frame_left.shape[1]
        cv2.line(combined, (separator_x, 0), (separator_x, combined.shape[0]), (255, 255, 255), 10)
        
        # --- Letterboxing Logic (Maintain Aspect Ratio) ---
        try:
            # Get current window size (returns x, y, w, h)
            _, _, win_w, win_h = cv2.getWindowImageRect('Tracking System')
        except:
            win_w, win_h = target_width, target_height

        # Safety check if window is minimized or not ready
        if win_w <= 0 or win_h <= 0:
            final_display = cv2.resize(combined, (target_width, target_height))
        else:
            # Calculate aspect ratios
            img_h, img_w = combined.shape[:2]
            img_aspect = img_w / img_h
            win_aspect = win_w / win_h
            
            if win_aspect > img_aspect:
                # Window is wider than image -> fit height, pad width (pillarbox)
                new_h = win_h
                new_w = int(win_h * img_aspect)
                # Ensure dimensions are valid
                if new_w <= 0: new_w = 1
                
                resized_content = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Calculate padding
                pad_left = (win_w - new_w) // 2
                pad_right = win_w - new_w - pad_left
                
                final_display = cv2.copyMakeBorder(resized_content, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                # Window is taller than image -> fit width, pad height (letterbox)
                new_w = win_w
                new_h = int(win_w / img_aspect)
                if new_h <= 0: new_h = 1
                
                resized_content = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                pad_top = (win_h - new_h) // 2
                pad_bottom = win_h - new_h - pad_top
                
                final_display = cv2.copyMakeBorder(resized_content, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        cv2.imshow('Tracking System', final_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        gray_prev = gray.copy()
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()