"""
Enhanced bottle tracking with side-by-side comparison.
Features a file selection dialog and detailed UI feedback.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
import tkinter as tk
from tkinter import filedialog, messagebox

TARGET_STREAM_WIDTH = 640
UI_SCALE = 0.75


def compute_tracking_resize(frame_width, frame_height, target_width=TARGET_STREAM_WIDTH):
    if frame_width <= target_width:
        return 1.0, frame_width, frame_height
    scale = target_width / float(frame_width)
    target_height = max(1, int(round(frame_height * scale)))
    return scale, target_width, target_height


def resize_for_tracking(frame, tracking_width, tracking_height, scale):
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (tracking_width, tracking_height), interpolation=cv2.INTER_AREA)

def log_tracking_resize(label, frame_width, frame_height, tracking_width, tracking_height, scale):
    if scale >= 1.0:
        print(f"{label}: no downscale ({frame_width}x{frame_height})")
        return
    print(
        f"{label}: downscale {frame_width}x{frame_height} -> "
        f"{tracking_width}x{tracking_height} (scale {scale:.3f})"
    )


class FeatureBasedTracker:
    def __init__(self, model_path, init_frames=5, num_features=30, search_radius=350):
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.init_frames = init_frames
        self.num_features = num_features
        self.base_search_radius = search_radius
        self.tracking_scale = 1.0

        # Use ORB - faster and often finds more features than SIFT in textured areas
        self.orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_HARRIS_SCORE)

        # Also keep GFTT as backup
        self.gftt_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=15,
            blockSize=7
        )

        self.lk_params = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        )
        self.base_gating_threshold = 100.0  # Threshold for blocking YOLO updates (pixels)
        self.search_radius = self.base_search_radius
        self.gating_threshold = self.base_gating_threshold
        self.hybrid_update_interval_sec = 2.0
        self.hybrid_feature_alpha = 0.25
        self.hybrid_yolo_alpha = 0.25
        self.base_hybrid_yolo_stable_max_delta = 25.0
        self.hybrid_yolo_stable_max_delta = self.base_hybrid_yolo_stable_max_delta
        self.hybrid_yolo_stable_frames = 3
        self.fps = 30.0
        self.hybrid_update_interval_frames = max(1, int(round(self.hybrid_update_interval_sec * self.fps)))
        self.apply_tracking_scale(self.tracking_scale)
        self.reset()

    def apply_tracking_scale(self, scale):
        if not scale or scale <= 0:
            scale = 1.0
        self.tracking_scale = float(scale)
        self.search_radius = max(1, int(round(self.base_search_radius * self.tracking_scale)))
        self.gating_threshold = max(1.0, float(self.base_gating_threshold * self.tracking_scale))
        self.hybrid_yolo_stable_max_delta = max(1.0, float(self.base_hybrid_yolo_stable_max_delta * self.tracking_scale))

    def adjust_search_radius(self, delta):
        self.base_search_radius = max(50, self.base_search_radius + delta)
        self.apply_tracking_scale(self.tracking_scale)

    def adjust_gating_threshold(self, delta):
        self.base_gating_threshold = max(10, self.base_gating_threshold + delta)
        self.apply_tracking_scale(self.tracking_scale)

    def set_fps(self, fps):
        if fps and fps > 0:
            self.fps = float(fps)
        else:
            self.fps = 30.0
        self.hybrid_update_interval_frames = max(1, int(round(self.hybrid_update_interval_sec * self.fps)))

    def smooth_center(self, current, target, alpha):
        if target is None:
            return current
        target_arr = np.array(target, dtype=np.float32)
        if current is None:
            return target_arr
        return current + (target_arr - current) * alpha

    def update_vectors_from_yolo(self, yolo_center):
        yolo_center_arr = np.array(yolo_center)
        for p in self.tracked_points:
            if p['active']:
                direction = yolo_center_arr - p['point']
                dist = np.linalg.norm(direction)
                if dist > 10:
                    p['direction'] = direction / dist
                    p['distance'] = dist

    def reset(self):
        self.frame_count = 0
        self.prev_gray = None
        self.tracked_points = []
        self.last_center = None
        self.last_bbox = None
        self.center_history = deque(maxlen=10)
        self.error_history = deque(maxlen=30)
        self.features_lost_total = 0
        self.intersections_used = 0
        self.last_yolo_center = None
        self.last_yolo_bbox = None
        self.last_yolo_frame = -999999
        self.frames_since_yolo = 0
        self.memory_center = None
        self.hybrid_last_update_frame = -999999
        self.hybrid_yolo_stable_count = 0
        # Wait to initialize features until the full search radius is in-frame.
        self.initialized = False
        self.init_ready_frames = 0

    def is_search_radius_in_frame(self, center, shape):
        if center is None:
            return False
        h, w = shape
        cx, cy = center
        r = self.search_radius
        return (
            (cx - r >= 0) and (cx + r < w) and
            (cy - r >= 0) and (cy + r < h)
        )

    def detect_with_yolo(self, frame):
        results = self.model(frame, verbose=False, conf=0.3)
        best_box = None
        best_conf = 0
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box.xyxy[0].cpu().numpy()
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            return center, best_box, best_conf
        return None, None, 0

    def find_features_around_point(self, gray, center, bbox=None):
        """Find stable feature points around a center point."""
        h, w = gray.shape
        cx, cy = int(center[0]), int(center[1])

        # Create search mask - ring around center
        mask = np.zeros((h, w), dtype=np.uint8)

        # Large outer radius for searching
        outer_radius = self.search_radius
        cv2.circle(mask, (cx, cy), outer_radius, 255, -1)

        # If we have bbox, exclude the object area
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            margin = 20
            cv2.rectangle(mask,
                          (max(0, x1 - margin), max(0, y1 - margin)),
                          (min(w, x2 + margin), min(h, y2 + margin)),
                          0, -1)
        else:
            # Exclude small area around center
            cv2.circle(mask, (cx, cy), 50, 0, -1)

        # Try ORB first
        keypoints = self.orb.detect(gray, mask)

        # If ORB didn't find enough, use Good Features to Track
        if len(keypoints) < self.num_features:
            corners = cv2.goodFeaturesToTrack(gray, mask=mask, **self.gftt_params)
            if corners is not None:
                for corner in corners:
                    pt = corner.ravel()
                    # Create a fake keypoint
                    kp = cv2.KeyPoint(pt[0], pt[1], 10)
                    keypoints.append(kp)

        if not keypoints:
            return []

        # Sort by response and take best ones, but spread them out
        keypoints = sorted(keypoints, key=lambda k: k.response if hasattr(k, 'response') and k.response else 0, reverse=True)

        # Select features that are well distributed
        selected_points = []
        min_dist_between = 30  # Minimum distance between features

        for kp in keypoints:
            pt = np.array([kp.pt[0], kp.pt[1]], dtype=np.float32)

            # Check distance from already selected points
            too_close = False
            for existing in selected_points:
                if np.linalg.norm(pt - existing['point']) < min_dist_between:
                    too_close = True
                    break

            if not too_close:
                direction = np.array(center) - pt
                dist = np.linalg.norm(direction)
                if dist > 10:  # Avoid points too close to center
                    direction = direction / dist
                    selected_points.append({
                        'point': pt,
                        'direction': direction,
                        'distance': dist,
                        'active': True,
                        'lost_count': 0,
                        'age': 0
                    })

            if len(selected_points) >= self.num_features:
                break

        return selected_points

    def track_points_optical_flow(self, prev_gray, curr_gray):
        """Track feature points using Lucas-Kanade optical flow."""
        if not self.tracked_points:
            return

        pts = np.array([p['point'] for p in self.tracked_points], dtype=np.float32)
        pts = pts.reshape(-1, 1, 2)

        new_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **self.lk_params
        )

        # Backward check for robustness
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, new_pts, None, **self.lk_params
        )

        h, w = curr_gray.shape

        for i, p in enumerate(self.tracked_points):
            new_pt = new_pts[i].ravel()

            # Check if point is still in frame
            in_frame = (10 < new_pt[0] < w - 10) and (10 < new_pt[1] < h - 10)

            if status[i][0] == 1 and back_status[i][0] == 1 and in_frame:
                fb_error = np.linalg.norm(pts[i].ravel() - back_pts[i].ravel())
                if fb_error < 3.0:  # Good tracking
                    p['point'] = new_pt
                    p['lost_count'] = 0
                    p['age'] += 1
                else:
                    p['lost_count'] += 1
            else:
                p['lost_count'] += 1

            # Mark as inactive if lost
            if p['lost_count'] > 2:
                p['active'] = False
                self.features_lost_total += 1

        # Remove inactive points
        self.tracked_points = [p for p in self.tracked_points if p['active']]

    def add_new_features(self, gray, estimated_center):
        """Add new features if we're running low."""
        if len(self.tracked_points) >= self.num_features // 2:
            return  # We have enough

        # Find new features around estimated center
        new_features = self.find_features_around_point(gray, estimated_center, self.last_bbox)

        # Only add features that aren't too close to existing ones
        min_dist = 40
        for new_f in new_features:
            too_close = False
            for existing in self.tracked_points:
                if np.linalg.norm(new_f['point'] - existing['point']) < min_dist:
                    too_close = True
                    break

            if not too_close:
                self.tracked_points.append(new_f)

            if len(self.tracked_points) >= self.num_features:
                break

    def estimate_center_from_intersections(self):
        """Estimate object center using line intersections from tracked points."""
        active = [p for p in self.tracked_points if p['active'] and p['lost_count'] == 0]

        if len(active) < 2:
            return self.last_center

        # Method 1: Direct estimates from each point
        direct_estimates = []
        for p in active:
            est = p['point'] + p['direction'] * p['distance']
            direct_estimates.append(est)

        # Method 2: Pairwise line intersections
        intersections = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                p1 = active[i]
                p2 = active[j]
                d1 = p1['direction']
                d2 = p2['direction']

                cross = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(cross) < 0.05:  # Nearly parallel
                    continue

                dp = p2['point'] - p1['point']
                t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross

                if t > 0:  # Forward intersection
                    intersection = p1['point'] + t * d1

                    # Sanity check - intersection shouldn't be too far from direct estimates
                    if direct_estimates:
                        avg_direct = np.mean(direct_estimates, axis=0)
                        if np.linalg.norm(intersection - avg_direct) < 200:
                            intersections.append(intersection)

        self.intersections_used = len(intersections)

        # Combine estimates
        all_estimates = direct_estimates + intersections

        if not all_estimates:
            return self.last_center

        # Use median for robustness against outliers
        all_estimates = np.array(all_estimates)
        center_x = np.median(all_estimates[:, 0])
        center_y = np.median(all_estimates[:, 1])

        center = np.array([center_x, center_y])

        # Smooth with history
        self.center_history.append(center)
        if len(self.center_history) >= 2:
            center = np.mean(self.center_history, axis=0)

        self.last_center = center
        return center

    def process_frame(self, frame, update_features_with_yolo=False):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        is_init_phase = not self.initialized
        is_gated = False

        # Always run YOLO to get ground truth for comparison
        prev_yolo_center = self.last_yolo_center
        prev_yolo_frame = self.last_yolo_frame
        yolo_center, yolo_bbox, yolo_conf = self.detect_with_yolo(frame)

        if yolo_center is not None:
            self.last_yolo_center = yolo_center
            self.last_yolo_bbox = yolo_bbox
            self.frames_since_yolo = 0
            self.last_yolo_frame = self.frame_count
        else:
            self.frames_since_yolo += 1

        if is_init_phase:
            # Initialization phase - use YOLO and build feature set
            if yolo_center is not None:
                self.last_center = np.array(yolo_center)
                self.last_bbox = yolo_bbox

                # Check if full search radius is in frame
                is_fully_in_frame = self.is_search_radius_in_frame(yolo_center, gray.shape)

                if not is_fully_in_frame:
                    print(f"Frame {self.frame_count}: Object too close to edge, waiting to init features...")
                    self.init_ready_frames = 0
                    # Avoid one-sided initialization when the search radius is clipped.
                    self.tracked_points = []
                else:
                    self.init_ready_frames += 1
                    # Track existing points
                    if self.prev_gray is not None and self.tracked_points:
                        self.track_points_optical_flow(self.prev_gray, gray)

                    # Find new features
                    new_features = self.find_features_around_point(gray, yolo_center, yolo_bbox)

                    # Replace or add features
                    if self.frame_count == 1 or len(self.tracked_points) < self.num_features // 3:
                        self.tracked_points = new_features
                    else:
                        # Add new features to fill gaps
                        self.add_new_features(gray, yolo_center)

                    print(f"Frame {self.frame_count}: {len(self.tracked_points)} features")
                    if self.init_ready_frames >= self.init_frames:
                        self.initialized = True
                        self.hybrid_last_update_frame = self.frame_count

            feature_center = yolo_center
        else:
            # Tracking phase - features only (no YOLO for position)
            if self.prev_gray is not None:
                self.track_points_optical_flow(self.prev_gray, gray)

            feature_center = self.estimate_center_from_intersections()

            # Try to add new features if we're running low
            if feature_center is not None and len(self.tracked_points) < self.num_features // 3:
                self.add_new_features(gray, feature_center)

            if feature_center is not None:
                self.memory_center = self.smooth_center(self.memory_center, feature_center, self.hybrid_feature_alpha)

            # Update feature point directions/distances based on YOLO when available
            if update_features_with_yolo and yolo_center is not None:
                ref_center = feature_center if feature_center is not None else self.memory_center
                # Check for gating (large discrepancy)
                if ref_center is not None:
                    dist = np.linalg.norm(np.array(yolo_center) - ref_center)
                    if dist > self.gating_threshold:
                        is_gated = True
                
                if not is_gated:
                    self.update_vectors_from_yolo(yolo_center)

            ref_center = feature_center if feature_center is not None else self.memory_center
            hybrid_is_gated = False
            if yolo_center is not None and ref_center is not None:
                dist = np.linalg.norm(np.array(yolo_center) - ref_center)
                if dist > self.gating_threshold:
                    hybrid_is_gated = True

            stable_this_frame = False
            if yolo_center is not None and prev_yolo_center is not None:
                if prev_yolo_frame == self.frame_count - 1:
                    delta = np.linalg.norm(np.array(yolo_center) - np.array(prev_yolo_center))
                    if delta <= self.hybrid_yolo_stable_max_delta:
                        stable_this_frame = True

            if yolo_center is None or hybrid_is_gated:
                self.hybrid_yolo_stable_count = 0
            elif stable_this_frame:
                self.hybrid_yolo_stable_count += 1
            else:
                self.hybrid_yolo_stable_count = 0

            should_hybrid_update = (
                yolo_center is not None
                and self.hybrid_update_interval_frames
                and (self.frame_count - self.hybrid_last_update_frame) >= self.hybrid_update_interval_frames
                and self.hybrid_yolo_stable_count >= self.hybrid_yolo_stable_frames
            )
            if should_hybrid_update:
                if ref_center is not None and not hybrid_is_gated:
                    self.memory_center = self.smooth_center(self.memory_center, yolo_center, self.hybrid_yolo_alpha)
                    self.update_vectors_from_yolo(yolo_center)
                    self.hybrid_last_update_frame = self.frame_count
            is_gated = is_gated or hybrid_is_gated
        if is_init_phase and feature_center is not None:
            self.memory_center = np.array(feature_center, dtype=np.float32)

        self.prev_gray = gray.copy()

        # Calculate error vs YOLO (for display purposes)
        if feature_center is not None and yolo_center is not None:
            error = np.sqrt((feature_center[0] - yolo_center[0])**2 +
                           (feature_center[1] - yolo_center[1])**2)
            if self.tracking_scale > 0:
                error = error / self.tracking_scale
            self.error_history.append(error)

        return feature_center, yolo_center, yolo_bbox, yolo_conf, is_init_phase, is_gated

    def get_stats(self):
        active = sum(1 for p in self.tracked_points if p['active'] and p['lost_count'] == 0)
        avg_error = np.mean(self.error_history) if self.error_history else 0
        current_error = self.error_history[-1] if self.error_history else 0
        return {
            'frame': self.frame_count,
            'active_features': active,
            'total_features': len(self.tracked_points),
            'intersections': self.intersections_used,
            'avg_error': avg_error,
            'current_error': current_error,
            'is_init': not self.initialized
        }


class UltralyticsTracker:
    def __init__(self, model_path, tracker_yaml):
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.tracker_yaml = tracker_yaml
        self.frame_count = 0

    def reset(self):
        self.frame_count = 0
        self.model.predictor = None

    def set_tracker_yaml(self, tracker_yaml):
        self.tracker_yaml = tracker_yaml
        self.model.predictor = None

    def process_frame(self, frame):
        self.frame_count += 1
        results = self.model.track(frame, persist=True, tracker=self.tracker_yaml, verbose=False)
        return results[0] if results else None


class VideoSelector:
    def __init__(self, default_dir):
        self.default_dir = default_dir

    def select_video(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            initialdir=self.default_dir,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return file_path if file_path else None


TRACKING_MODE_FEATURE = "Feature-based (custom)"
TRACKING_MODE_ULTRA = "Ultralytics .track"
ULTRALYTICS_TRACKERS = {
    "ByteTrack": "bytetrack.yaml",
    "BoT-SORT": "botsort.yaml",
}
CUSTOM_TRACKER_FILES = {
    "ByteTrack": "bytetrack.custom.yaml",
    "BoT-SORT": "botsort.custom.yaml",
}


def get_custom_tracker_dir(script_dir):
    custom_dir = os.path.join(script_dir, "tmp", "trackers")
    os.makedirs(custom_dir, exist_ok=True)
    return custom_dir


def get_custom_tracker_path(script_dir, tracker_label):
    file_name = CUSTOM_TRACKER_FILES.get(tracker_label, "tracker.custom.yaml")
    return os.path.join(get_custom_tracker_dir(script_dir), file_name)


def get_default_tracker_path(tracker_label):
    from ultralytics.utils.checks import check_yaml

    tracker_yaml = ULTRALYTICS_TRACKERS.get(tracker_label, "bytetrack.yaml")
    return check_yaml(tracker_yaml, hard=True)


def get_active_tracker_path(script_dir, tracker_label):
    custom_path = get_custom_tracker_path(script_dir, tracker_label)
    if os.path.exists(custom_path):
        return custom_path
    return get_default_tracker_path(tracker_label)


def read_tracker_yaml(script_dir, tracker_label, prefer_custom=True):
    default_path = get_default_tracker_path(tracker_label)
    custom_path = get_custom_tracker_path(script_dir, tracker_label)
    use_custom = prefer_custom and os.path.exists(custom_path)
    path = custom_path if use_custom else default_path
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return path, content, use_custom


def validate_yaml_text(text):
    try:
        import yaml

        yaml.safe_load(text)
    except Exception as exc:
        return str(exc)
    return None


class TrackingConfigDialog:
    def __init__(self, default_dir):
        self.default_dir = default_dir
        self.result = None

    def select(self, default_video_path=None, default_mode=None, default_tracker=None):
        root = tk.Tk()
        root.title("Tracking Settings")
        root.resizable(False, False)

        default_video_path = default_video_path or ""
        if default_mode not in (TRACKING_MODE_FEATURE, TRACKING_MODE_ULTRA):
            default_mode = TRACKING_MODE_FEATURE
        if default_tracker not in ULTRALYTICS_TRACKERS:
            default_tracker = "ByteTrack"

        video_var = tk.StringVar(value=default_video_path)
        mode_var = tk.StringVar(value=default_mode)
        tracker_var = tk.StringVar(value=default_tracker)

        def browse():
            path = filedialog.askopenfilename(
                title="Select Video File",
                initialdir=self.default_dir,
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"),
                    ("All files", "*.*"),
                ],
            )
            if path:
                video_var.set(path)

        def update_tracker_state(*_):
            if mode_var.get() == TRACKING_MODE_ULTRA:
                tracker_menu.configure(state="normal")
            else:
                tracker_menu.configure(state="disabled")

        def on_ok():
            path = video_var.get().strip()
            if not path:
                messagebox.showerror("Missing video", "Please select a video file.")
                return
            self.result = {
                "video_path": path,
                "mode": mode_var.get(),
                "tracker": tracker_var.get(),
            }
            root.destroy()

        def on_cancel():
            self.result = None
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_cancel)

        container = tk.Frame(root, padx=12, pady=12)
        container.grid(row=0, column=0)

        tk.Label(container, text="Video file:").grid(row=0, column=0, sticky="w")
        tk.Entry(container, textvariable=video_var, width=50).grid(row=1, column=0, columnspan=2, pady=(4, 6))
        tk.Button(container, text="Browse...", command=browse).grid(row=1, column=2, padx=(6, 0))

        tk.Label(container, text="Tracking mode:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        mode_menu = tk.OptionMenu(container, mode_var, TRACKING_MODE_FEATURE, TRACKING_MODE_ULTRA)
        mode_menu.config(width=24)
        mode_menu.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 6))

        tk.Label(container, text="Ultralytics tracker:").grid(row=4, column=0, sticky="w", pady=(4, 0))
        tracker_menu = tk.OptionMenu(container, tracker_var, *ULTRALYTICS_TRACKERS.keys())
        tracker_menu.config(width=24)
        tracker_menu.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 10))

        buttons = tk.Frame(container)
        buttons.grid(row=6, column=0, columnspan=3, sticky="e")
        tk.Button(buttons, text="Cancel", command=on_cancel).grid(row=0, column=0, padx=(0, 6))
        tk.Button(buttons, text="OK", command=on_ok).grid(row=0, column=1)

        mode_var.trace_add("write", update_tracker_state)
        update_tracker_state()

        root.mainloop()
        return self.result


class TrackerParamsDialog:
    def __init__(self, script_dir):
        self.script_dir = script_dir
        self.result = None

    def select(self, current_tracker_label):
        root = tk.Tk()
        root.title("Ultralytics Tracker Parameters")
        root.geometry("760x520")

        if current_tracker_label not in ULTRALYTICS_TRACKERS:
            current_tracker_label = "ByteTrack"

        tracker_var = tk.StringVar(value=current_tracker_label)
        status_var = tk.StringVar(value="")
        dirty_var = {"value": False}
        last_tracker = {"value": tracker_var.get()}
        suppress_change = {"value": False}

        def load_for_tracker(use_defaults=False):
            label = tracker_var.get()
            if use_defaults:
                path = get_default_tracker_path(label)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                source = "default"
            else:
                path, content, is_custom = read_tracker_yaml(self.script_dir, label, prefer_custom=True)
                source = "custom" if is_custom else "default"
            text_box.delete("1.0", tk.END)
            text_box.insert("1.0", content)
            text_box.edit_modified(False)
            dirty_var["value"] = False
            status_var.set(f"Loaded {source}: {path}")

        def on_text_modified(event=None):
            if text_box.edit_modified():
                dirty_var["value"] = True
                text_box.edit_modified(False)

        def save_yaml():
            content = text_box.get("1.0", tk.END).rstrip()
            error = validate_yaml_text(content)
            if error:
                messagebox.showerror("Invalid YAML", f"YAML parse error:\n{error}")
                return None
            path = get_custom_tracker_path(self.script_dir, tracker_var.get())
            with open(path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            dirty_var["value"] = False
            status_var.set(f"Saved custom: {path}")
            return path

        def on_tracker_change(*_):
            if suppress_change["value"]:
                return
            if dirty_var["value"]:
                discard = messagebox.askyesno(
                    "Discard changes?",
                    "You have unsaved changes. Discard them and load another tracker?",
                )
                if not discard:
                    suppress_change["value"] = True
                    tracker_var.set(last_tracker["value"])
                    suppress_change["value"] = False
                    return
            last_tracker["value"] = tracker_var.get()
            load_for_tracker(use_defaults=False)

        def on_save():
            save_yaml()

        def on_apply():
            path = save_yaml()
            if not path:
                return
            self.result = {
                "tracker": tracker_var.get(),
                "yaml_path": path,
                "apply_to_current": tracker_var.get() == current_tracker_label,
            }
            root.destroy()

        def on_cancel():
            self.result = None
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_cancel)

        top = tk.Frame(root, padx=10, pady=10)
        top.pack(fill="both", expand=True)
        for col in range(4):
            top.columnconfigure(col, weight=1)

        tk.Label(top, text="Tracker:").grid(row=0, column=0, sticky="w")
        tracker_menu = tk.OptionMenu(top, tracker_var, *ULTRALYTICS_TRACKERS.keys())
        tracker_menu.config(width=20)
        tracker_menu.grid(row=0, column=1, sticky="w")

        tk.Button(top, text="Load Defaults", command=lambda: load_for_tracker(use_defaults=True)).grid(
            row=0, column=2, padx=6
        )
        tk.Button(top, text="Reload", command=lambda: load_for_tracker(use_defaults=False)).grid(row=0, column=3)

        text_frame = tk.Frame(top, bd=1, relief="sunken")
        text_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(10, 6))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        text_box = tk.Text(text_frame, wrap="none", undo=True)
        text_box.grid(row=0, column=0, sticky="nsew")
        text_box.bind("<<Modified>>", on_text_modified)

        y_scroll = tk.Scrollbar(text_frame, command=text_box.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        text_box.configure(yscrollcommand=y_scroll.set)

        x_scroll = tk.Scrollbar(text_frame, command=text_box.xview, orient="horizontal")
        x_scroll.grid(row=1, column=0, sticky="ew")
        text_box.configure(xscrollcommand=x_scroll.set)

        status = tk.Label(top, textvariable=status_var, anchor="w")
        status.grid(row=2, column=0, columnspan=4, sticky="w")

        hint = tk.Label(
            top,
            text="Save applies to current tracker if it matches; switch trackers via Settings (S) if needed.",
            anchor="w",
        )
        hint.grid(row=3, column=0, columnspan=4, sticky="w", pady=(2, 8))

        buttons = tk.Frame(top)
        buttons.grid(row=4, column=0, columnspan=4, sticky="e")
        tk.Button(buttons, text="Cancel", command=on_cancel).grid(row=0, column=0, padx=(0, 6))
        tk.Button(buttons, text="Save", command=on_save).grid(row=0, column=1, padx=(0, 6))
        tk.Button(buttons, text="Apply & Close", command=on_apply).grid(row=0, column=2)

        top.rowconfigure(1, weight=1)

        tracker_var.trace_add("write", on_tracker_change)
        load_for_tracker(use_defaults=False)

        root.mainloop()
        return self.result


def draw_feature_view(frame, tracker, feature_center, yolo_center, yolo_bbox, is_init_phase, scale, coord_scale=1.0):
    vis = frame.copy()
    visual_scale = scale * UI_SCALE
    pt_radius = max(4, int(8 * visual_scale))
    line_thick = max(1, int(2 * visual_scale))
    cross_size = max(14, int(26 * visual_scale))

    # Draw search radius
    if feature_center is not None:
        fc = tuple(map(int, np.array(feature_center) * coord_scale))
        radius = max(1, int(round(tracker.search_radius * coord_scale)))
        cv2.circle(vis, fc, radius, (60, 60, 60), 1, cv2.LINE_AA)

    # Draw each tracked feature
    for i, p in enumerate(tracker.tracked_points):
        if not p['active']:
            continue

        pt = tuple(map(int, p['point'] * coord_scale))

        # Color by age - older = more trusted
        if p['age'] > 20:
            color = (0, 255, 0)  # Green - well established
        elif p['age'] > 5:
            color = (0, 255, 255)  # Yellow - building trust
        else:
            color = (0, 165, 255)  # Orange - new

        # Draw point
        cv2.circle(vis, pt, pt_radius, color, -1)
        cv2.circle(vis, pt, pt_radius + 2, (255, 255, 255), 1)

        # Draw direction line
        line_len = min(p['distance'], 200)
        end_pt = p['point'] + p['direction'] * line_len
        end_pt_tuple = tuple(map(int, end_pt * coord_scale))
        cv2.arrowedLine(vis, pt, end_pt_tuple, (0, 255, 255), line_thick, cv2.LINE_AA, tipLength=0.06)

        # Draw point number
        cv2.putText(vis, str(i), (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw estimated center
    if feature_center is not None:
        fc = tuple(map(int, np.array(feature_center) * coord_scale))
        cv2.line(vis, (fc[0] - cross_size, fc[1]), (fc[0] + cross_size, fc[1]), (255, 0, 255), line_thick + 1)
        cv2.line(vis, (fc[0], fc[1] - cross_size), (fc[0], fc[1] + cross_size), (255, 0, 255), line_thick + 1)
        cv2.circle(vis, fc, int(cross_size * 0.5), (255, 0, 255), line_thick + 1)
        cv2.circle(vis, fc, max(2, int(5 * visual_scale)), (255, 0, 255), -1)

    # Draw YOLO center for comparison (when tracking)
    if yolo_center is not None and not is_init_phase:
        yc = tuple(map(int, np.array(yolo_center) * coord_scale))
        cv2.circle(vis, yc, pt_radius, (0, 0, 255), line_thick)
        # Draw error line
        if feature_center is not None:
            fc = tuple(map(int, np.array(feature_center) * coord_scale))
            cv2.line(vis, fc, yc, (0, 0, 255), 1, cv2.LINE_AA)

    return vis


def draw_yolo_view(frame, yolo_center, yolo_bbox, yolo_conf, scale, coord_scale=1.0):
    vis = frame.copy()
    visual_scale = scale * UI_SCALE
    line_thick = max(1, int(2 * visual_scale))
    pt_radius = max(6, int(10 * visual_scale))
    cross_size = max(14, int(24 * visual_scale))

    if yolo_bbox is not None:
        x1, y1, x2, y2 = [int(round(v * coord_scale)) for v in yolo_bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), line_thick + 1)

    if yolo_center is not None:
        yc = tuple(map(int, np.array(yolo_center) * coord_scale))
        cv2.circle(vis, yc, pt_radius, (0, 0, 255), -1)
        cv2.circle(vis, yc, pt_radius + 2, (255, 255, 255), line_thick)
        cv2.line(vis, (yc[0] - cross_size, yc[1]), (yc[0] + cross_size, yc[1]), (0, 0, 255), line_thick)
        cv2.line(vis, (yc[0], yc[1] - cross_size), (yc[0], yc[1] + cross_size), (0, 0, 255), line_thick)

    return vis


def draw_hybrid_view(frame, memory_center, yolo_center, scale, is_gated=False, coord_scale=1.0):
    """
    Draw hybrid view with a stable memory crosshair.
    - Pink: Internal memory center (always primary).
    - Small Red dot: The 'bad' YOLO detection being ignored.
    """
    vis = frame.copy()
    visual_scale = scale * UI_SCALE
    line_thick = max(1, int(2 * visual_scale))
    pt_radius = max(6, int(10 * visual_scale))
    cross_size = max(14, int(24 * visual_scale))

    if memory_center is None:
        return vis, False

    primary_center = tuple(map(int, np.array(memory_center) * coord_scale))
    primary_color = (255, 0, 255)  # Pink

    # Draw Primary Crosshair
    cv2.circle(vis, primary_center, pt_radius, primary_color, -1)
    cv2.circle(vis, primary_center, pt_radius + 2, (255, 255, 255), line_thick)
    cv2.line(vis, (primary_center[0] - cross_size, primary_center[1]), (primary_center[0] + cross_size, primary_center[1]), primary_color, line_thick)
    cv2.line(vis, (primary_center[0], primary_center[1] - cross_size), (primary_center[0], primary_center[1] + cross_size), primary_color, line_thick)

    # If Gated, show the 'Bad' YOLO detection as a reference
    if is_gated and yolo_center is not None:
        yc = tuple(map(int, np.array(yolo_center) * coord_scale))
        # Draw small red dot for the bad detection
        cv2.circle(vis, yc, int(pt_radius * 0.5), (0, 0, 255), -1)
        cv2.circle(vis, yc, int(pt_radius * 0.5) + 2, (255, 255, 255), 1)
        # Draw "GATED" text near the bad detection
        cv2.putText(vis, "BLOCKED", (yc[0] + 15, yc[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # Draw a dashed-style line between them to show the deviation
        cv2.line(vis, primary_center, yc, (0, 0, 255), 1, cv2.LINE_AA)

    return vis, False


def track_color(track_id):
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
        (128, 255, 0),
    ]
    if track_id is None:
        return (0, 0, 255)
    return palette[track_id % len(palette)]


def extract_tracks(result, coord_scale=1.0):
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.cpu().numpy() if boxes.id is not None else None
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    tracks = []
    for i in range(len(xyxy)):
        track_id = int(ids[i]) if ids is not None else None
        conf = float(confs[i]) if confs is not None else 0.0
        box = xyxy[i]
        if coord_scale != 1.0:
            box = box * coord_scale
        tracks.append((box, track_id, conf))
    return tracks


def draw_track_view(frame, tracks, scale, tracker_label):
    vis = frame.copy()
    visual_scale = scale * UI_SCALE
    line_thick = max(1, int(2 * visual_scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, track_id, conf in tracks:
        x1, y1, x2, y2 = map(int, box)
        color = track_color(track_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_thick)
        label = f"ID {track_id}" if track_id is not None else "ID ?"
        if conf > 0:
            label += f" {conf:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 8)), font, 0.5, color, 1)

    if tracker_label:
        cv2.putText(vis, tracker_label, (10, 30), font, 0.9, (0, 255, 255), 2)

    return vis


def create_track_display(track_vis, frame_index, total_frames, is_paused, speed_multiplier, display_width, track_count, tracker_label):
    orig_h, orig_w = track_vis.shape[:2]
    video_scale = display_width / orig_w
    video_height = int(orig_h * video_scale)
    video_resized = cv2.resize(track_vis, (display_width, video_height))

    bar_height = 50
    bar = np.zeros((bar_height, display_width, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)

    font = cv2.FONT_HERSHEY_SIMPLEX
    status_text = "PAUSED" if is_paused else "PLAYING"
    status_color = (0, 165, 255) if is_paused else (0, 255, 0)
    cv2.putText(bar, status_text, (10, 32), font, 0.7, status_color, 2)

    cv2.putText(bar, "TRACK", (120, 32), font, 0.7, (0, 255, 255), 2)
    cv2.putText(bar, f"F:{frame_index}/{total_frames}", (200, 32), font, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, f"Tracks:{track_count}", (360, 32), font, 0.7, (0, 255, 0), 2)
    if tracker_label:
        cv2.putText(bar, f"Alg:{tracker_label}", (520, 32), font, 0.6, (255, 255, 255), 1)

    prog_x = 1030
    prog_w = display_width - prog_x - 10
    if prog_w > 50:
        progress = frame_index / total_frames if total_frames > 0 else 0
        cv2.rectangle(bar, (prog_x, 18), (prog_x + prog_w, 28), (80, 80, 80), -1)
        cv2.rectangle(bar, (prog_x, 18), (prog_x + int(prog_w * progress), 28), (0, 255, 0), -1)

    ctrl_height = 25
    ctrl_bar = np.zeros((ctrl_height, display_width, 3), dtype=np.uint8)
    ctrl_bar[:] = (30, 30, 30)
    controls = "SPACE:Play/Pause  R:Reset  O:Open  S:Settings  P:Params  +/-:Speed  Scroll:Size  Q:Quit"
    cv2.putText(ctrl_bar, controls, (10, 18), font, 0.45, (120, 120, 120), 1)

    result = np.vstack([bar, video_resized, ctrl_bar])
    return result


def create_display(yolo_vis, feature_vis, tracker, yolo_conf, total_frames, is_paused, speed_multiplier, display_width, view_mode, hybrid_vis=None, using_yolo=False, update_features_with_yolo=False, is_gated=False, show_tracking_view=False):
    orig_h, orig_w = yolo_vis.shape[:2]

    if view_mode == 0:
        video_width = display_width // 2
    else:
        video_width = display_width

    video_scale = video_width / orig_w
    video_height = int(orig_h * video_scale)

    if view_mode == 0:
        yolo_resized = cv2.resize(yolo_vis, (video_width, video_height))
        feature_resized = cv2.resize(feature_vis, (video_width, video_height))
        cv2.putText(yolo_resized, "YOLO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(feature_resized, "FEATURES", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        video_combined = np.hstack([yolo_resized, feature_resized])
    elif view_mode == 1:
        video_combined = cv2.resize(yolo_vis, (video_width, video_height))
        cv2.putText(video_combined, "YOLO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    elif view_mode == 2:
        video_combined = cv2.resize(feature_vis, (video_width, video_height))
        cv2.putText(video_combined, "FEATURES", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        # Hybrid mode (view_mode == 3)
        if hybrid_vis is not None:
            video_combined = cv2.resize(hybrid_vis, (video_width, video_height))
        else:
            video_combined = cv2.resize(yolo_vis, (video_width, video_height))
        # Label and color based on active method
        if using_yolo:
            cv2.putText(video_combined, "HYBRID", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(video_combined, "YOLO", (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(video_combined, "HYBRID", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            cv2.putText(video_combined, "TRACKING", (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Info bar
    stats = tracker.get_stats()
    bar_height = 50
    bar = np.zeros((bar_height, display_width, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Status
    status_text = "PAUSED" if is_paused else "PLAYING"
    status_color = (0, 165, 255) if is_paused else (0, 255, 0)
    cv2.putText(bar, status_text, (10, 32), font, 0.7, status_color, 2)

    # Mode
    mode_text = "INIT" if stats['is_init'] else "TRACK"
    mode_color = (0, 255, 255) if stats['is_init'] else (0, 255, 0)
    cv2.putText(bar, mode_text, (120, 32), font, 0.7, mode_color, 2)

    # Frame
    cv2.putText(bar, f"F:{stats['frame']}/{total_frames}", (200, 32), font, 0.6, (255, 255, 255), 1)

    # Features - more prominent
    feat_count = stats['active_features']
    if feat_count >= 15:
        fc = (0, 255, 0)
    elif feat_count >= 8:
        fc = (0, 255, 255)
    elif feat_count >= 3:
        fc = (0, 165, 255)
    else:
        fc = (0, 0, 255)
    cv2.putText(bar, f"Feat:{feat_count}", (360, 32), font, 0.7, fc, 2)

    # Intersections
    cv2.putText(bar, f"Int:{stats['intersections']}", (480, 32), font, 0.6, (255, 255, 0), 1)

    # Error
    if stats['current_error'] > 0:
        err = stats['current_error']
        if err < 20:
            ec = (0, 255, 0)
        elif err < 50:
            ec = (0, 255, 255)
        else:
            ec = (0, 0, 255)
        cv2.putText(bar, f"Err:{err:.0f}px", (580, 32), font, 0.6, ec, 1)

    # Speed
    cv2.putText(bar, f"{speed_multiplier:.1f}x", (700, 32), font, 0.6, (255, 255, 255), 1)

    # View
    view_names = ["BOTH", "YOLO", "FEAT", "HYBRID"]
    view_color = (255, 255, 0)
    if view_mode == 3:
        view_color = (0, 255, 0) if using_yolo else (0, 165, 255)
    cv2.putText(bar, f"V:{view_names[view_mode]}", (770, 32), font, 0.6, view_color, 1)

    # Update mode indicator & Gating Status
    upd_text = "UPD:ON" if update_features_with_yolo else "UPD:OFF"
    upd_color = (0, 255, 0) if update_features_with_yolo else (100, 100, 100)
    src_text = "SRC:DS" if show_tracking_view else "SRC:ORIG"
    cv2.putText(bar, src_text, (840, 32), font, 0.5, (180, 180, 180), 1)
    cv2.putText(bar, upd_text, (930, 32), font, 0.5, upd_color, 1)
    
    # Gating Threshold + Search Radius
    gate_color = (0, 255, 255)
    if is_gated:
        gate_color = (0, 0, 255) # RED if gated
        cv2.putText(bar, "GATED!", (1100, 32), font, 0.6, (0, 0, 255), 2)
    
    cv2.putText(
        bar,
        f"Gate:{tracker.base_gating_threshold:.0f}px Rad:{tracker.base_search_radius:.0f}px",
        (1000, 32),
        font,
        0.5,
        gate_color,
        1,
    )

    # Progress bar
    prog_x = 1180
    prog_w = display_width - prog_x - 10
    if prog_w > 50:
        progress = stats['frame'] / total_frames if total_frames > 0 else 0
        cv2.rectangle(bar, (prog_x, 18), (prog_x + prog_w, 28), (80, 80, 80), -1)
        cv2.rectangle(bar, (prog_x, 18), (prog_x + int(prog_w * progress), 28), (0, 255, 0), -1)

    # Controls bar
    ctrl_height = 25
    ctrl_bar = np.zeros((ctrl_height, display_width, 3), dtype=np.uint8)
    ctrl_bar[:] = (30, 30, 30)
    controls = "SPACE:Play/Pause  V:View  U:Update  D:Downscale  [ / ]:Gate Thresh  , / .:Radius  R:Reset  O:Open  S:Settings  +/-:Speed  Q:Quit"
    cv2.putText(ctrl_bar, controls, (10, 18), font, 0.45, (120, 120, 120), 1)

    result = np.vstack([bar, video_combined, ctrl_bar])
    return result


def run_feature_tracking(video_path, model_path, selector, config_dialog, tracker_choice):
    print(f"Video: {os.path.basename(video_path)}")

    tracker = FeatureBasedTracker(model_path, init_frames=5, num_features=30, search_radius=350)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker.set_fps(fps)
    tracking_scale, tracking_width, tracking_height = compute_tracking_resize(frame_width, frame_height)
    tracker.apply_tracking_scale(tracking_scale)
    tracking_to_display_scale = 1.0 / tracking_scale if tracking_scale > 0 else 1.0
    log_tracking_resize("Tracking input", frame_width, frame_height, tracking_width, tracking_height, tracking_scale)

    print(f"FPS: {fps:.1f}, Frames: {total_frames}")
    print("\nSPACE:Play/Pause V:View U:Update D:Downscale [ / ]:Gate Thresh , / .:Radius R:Reset O:Open S:Settings Q:Quit")

    display_width = 1200
    min_width = 600
    max_width = 1800
    view_mode = 0

    window_name = "Bottle Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def mouse_callback(event, x, y, flags, param):
        nonlocal display_width
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                display_width = min(max_width, display_width + 100)
            else:
                display_width = max(min_width, display_width - 100)

    cv2.setMouseCallback(window_name, mouse_callback)

    paused = True
    base_delay = max(1, int(1000 / fps)) if fps > 0 else 33
    delay = base_delay
    speed_multiplier = 1.0

    last_yolo_vis = None
    last_feature_vis = None
    last_hybrid_vis = None
    last_using_yolo = False
    last_yolo_conf = 0
    last_scale = frame_width / 1920
    last_feature_center = None
    last_yolo_center = None
    last_yolo_bbox = None
    last_memory_center = None
    last_is_init = True
    update_features_with_yolo = False
    last_is_gated = False
    show_tracking_view = False
    last_frame = None
    last_tracking_frame = None

    def render_views():
        nonlocal last_yolo_vis, last_feature_vis, last_hybrid_vis, last_using_yolo
        if last_frame is None or last_tracking_frame is None:
            return
        base_frame = last_tracking_frame if show_tracking_view else last_frame
        coord_scale = 1.0 if show_tracking_view else tracking_to_display_scale
        view_scale = (tracking_width if show_tracking_view else frame_width) / 1920
        last_yolo_vis = draw_yolo_view(base_frame, last_yolo_center, last_yolo_bbox, last_yolo_conf, view_scale, coord_scale)
        last_feature_vis = draw_feature_view(base_frame, tracker, last_feature_center, last_yolo_center, last_yolo_bbox, last_is_init, view_scale, coord_scale)
        last_hybrid_vis, last_using_yolo = draw_hybrid_view(base_frame, last_memory_center, last_yolo_center, view_scale, last_is_gated, coord_scale)

    ret, frame = cap.read()
    if ret:
        tracking_frame = resize_for_tracking(frame, tracking_width, tracking_height, tracking_scale)
        feature_center, yolo_center, yolo_bbox, yolo_conf, is_init, is_gated = tracker.process_frame(tracking_frame, update_features_with_yolo)
        last_yolo_conf = yolo_conf
        last_feature_center = feature_center
        last_yolo_center = yolo_center
        last_yolo_bbox = yolo_bbox
        last_memory_center = tracker.memory_center
        last_is_init = is_init
        last_is_gated = is_gated
        last_frame = frame
        last_tracking_frame = tracking_frame
        render_views()

    next_selection = None
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker.reset()
                continue

            tracking_frame = resize_for_tracking(frame, tracking_width, tracking_height, tracking_scale)
            feature_center, yolo_center, yolo_bbox, yolo_conf, is_init, is_gated = tracker.process_frame(tracking_frame, update_features_with_yolo)
            last_yolo_conf = yolo_conf
            last_feature_center = feature_center
            last_yolo_center = yolo_center
            last_yolo_bbox = yolo_bbox
            last_memory_center = tracker.memory_center
            last_is_init = is_init
            last_is_gated = is_gated
            last_frame = frame
            last_tracking_frame = tracking_frame
            render_views()

        if last_yolo_vis is not None and last_feature_vis is not None:
            display = create_display(
                last_yolo_vis, last_feature_vis, tracker, last_yolo_conf,
                total_frames, paused, speed_multiplier, display_width, view_mode,
                last_hybrid_vis, last_using_yolo, update_features_with_yolo, last_is_gated, show_tracking_view
            )
            cv2.imshow(window_name, display)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('v'):
            view_mode = (view_mode + 1) % 4
        elif key == ord('u'):
            update_features_with_yolo = not update_features_with_yolo
            print(f"Update features with YOLO: {'ON' if update_features_with_yolo else 'OFF'}")
        elif key == ord('d'):
            show_tracking_view = not show_tracking_view
            print(f"Display source: {'DOWNSCALED' if show_tracking_view else 'ORIGINAL'}")
            render_views()
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker.reset()
            tracker.apply_tracking_scale(tracking_scale)
            paused = True
            ret, frame = cap.read()
            if ret:
                tracking_frame = resize_for_tracking(frame, tracking_width, tracking_height, tracking_scale)
                feature_center, yolo_center, yolo_bbox, yolo_conf, is_init, is_gated = tracker.process_frame(tracking_frame, update_features_with_yolo)
                last_yolo_conf = yolo_conf
                last_feature_center = feature_center
                last_yolo_center = yolo_center
                last_yolo_bbox = yolo_bbox
                last_memory_center = tracker.memory_center
                last_is_init = is_init
                last_is_gated = is_gated
                last_frame = frame
                last_tracking_frame = tracking_frame
                render_views()
        elif key == ord('o'):
            new_video = selector.select_video()
            if new_video:
                cap.release()
                video_path = new_video
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                tracker.set_fps(fps)
                tracking_scale, tracking_width, tracking_height = compute_tracking_resize(frame_width, frame_height)
                tracking_to_display_scale = 1.0 / tracking_scale if tracking_scale > 0 else 1.0
                base_delay = max(1, int(1000 / fps)) if fps > 0 else 33
                delay = base_delay
                speed_multiplier = 1.0
                last_scale = frame_width / 1920
                tracker.reset()
                tracker.apply_tracking_scale(tracking_scale)
                log_tracking_resize("Tracking input", frame_width, frame_height, tracking_width, tracking_height, tracking_scale)
                paused = True
                ret, frame = cap.read()
                if ret:
                    tracking_frame = resize_for_tracking(frame, tracking_width, tracking_height, tracking_scale)
                    feature_center, yolo_center, yolo_bbox, yolo_conf, is_init, is_gated = tracker.process_frame(tracking_frame, update_features_with_yolo)
                    last_yolo_conf = yolo_conf
                    last_feature_center = feature_center
                    last_yolo_center = yolo_center
                    last_yolo_bbox = yolo_bbox
                    last_memory_center = tracker.memory_center
                    last_is_init = is_init
                    last_is_gated = is_gated
                    last_frame = frame
                    last_tracking_frame = tracking_frame
                    render_views()
        elif key == ord('s'):
            selection = config_dialog.select(
                default_video_path=video_path,
                default_mode=TRACKING_MODE_FEATURE,
                default_tracker=tracker_choice,
            )
            if selection:
                next_selection = selection
                break
        elif key == ord(']'):
            tracker.adjust_gating_threshold(10)
            print(
                f"Gating Threshold: {tracker.base_gating_threshold} "
                f"(scaled {tracker.gating_threshold:.0f})"
            )
        elif key == ord('['):
            tracker.adjust_gating_threshold(-10)
            print(
                f"Gating Threshold: {tracker.base_gating_threshold} "
                f"(scaled {tracker.gating_threshold:.0f})"
            )
        elif key == ord('.'):
            tracker.adjust_search_radius(10)
            print(
                f"Search radius: {tracker.base_search_radius} "
                f"(scaled {tracker.search_radius:.0f})"
            )
        elif key == ord(','):
            tracker.adjust_search_radius(-10)
            print(
                f"Search radius: {tracker.base_search_radius} "
                f"(scaled {tracker.search_radius:.0f})"
            )
        elif key == ord('+') or key == ord('='):
            delay = max(1, delay - 5)
            speed_multiplier = base_delay / delay
        elif key == ord('-'):
            delay = min(500, delay + 5)
            speed_multiplier = base_delay / delay

    cap.release()
    cv2.destroyAllWindows()
    return next_selection


def run_ultralytics_tracking(video_path, model_path, tracker_yaml, tracker_label, selector, config_dialog, script_dir):
    print(f"Video: {os.path.basename(video_path)}")

    tracker = UltralyticsTracker(model_path, tracker_yaml)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracking_scale, tracking_width, tracking_height = compute_tracking_resize(frame_width, frame_height)
    tracking_to_display_scale = 1.0 / tracking_scale if tracking_scale > 0 else 1.0

    print(f"FPS: {fps:.1f}, Frames: {total_frames}")
    log_tracking_resize("Tracking input", frame_width, frame_height, tracking_width, tracking_height, tracking_scale)
    print("\nSPACE:Play/Pause R:Reset O:Open S:Settings P:Params +/-:Speed Q:Quit")

    display_width = 1200
    min_width = 600
    max_width = 1800

    window_name = "Bottle Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def mouse_callback(event, x, y, flags, param):
        nonlocal display_width
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                display_width = min(max_width, display_width + 100)
            else:
                display_width = max(min_width, display_width - 100)

    cv2.setMouseCallback(window_name, mouse_callback)

    paused = True
    base_delay = max(1, int(1000 / fps)) if fps > 0 else 33
    delay = base_delay
    speed_multiplier = 1.0

    last_track_vis = None
    last_track_count = 0
    last_scale = frame_width / 1920

    ret, frame = cap.read()
    if ret:
        tracking_frame = resize_for_tracking(frame, tracking_width, tracking_height, tracking_scale)
        result = tracker.process_frame(tracking_frame)
        tracks = extract_tracks(result, tracking_to_display_scale)
        last_track_count = len(tracks)
        last_track_vis = draw_track_view(frame, tracks, last_scale, tracker_label)

    next_selection = None
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker.reset()
                continue

            tracking_frame = resize_for_tracking(frame, tracking_width, tracking_height, tracking_scale)
            result = tracker.process_frame(tracking_frame)
            tracks = extract_tracks(result, tracking_to_display_scale)
            last_track_count = len(tracks)
            last_track_vis = draw_track_view(frame, tracks, last_scale, tracker_label)

        if last_track_vis is not None:
            display = create_track_display(
                last_track_vis, tracker.frame_count, total_frames, paused,
                speed_multiplier, display_width, last_track_count, tracker_label
            )
            cv2.imshow(window_name, display)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker.reset()
            paused = True
            ret, frame = cap.read()
            if ret:
                tracking_frame = resize_for_tracking(frame, tracking_width, tracking_height, tracking_scale)
                result = tracker.process_frame(tracking_frame)
                tracks = extract_tracks(result, tracking_to_display_scale)
                last_track_count = len(tracks)
                last_track_vis = draw_track_view(frame, tracks, last_scale, tracker_label)
        elif key == ord('o'):
            new_video = selector.select_video()
            if new_video:
                cap.release()
                video_path = new_video
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                tracking_scale, tracking_width, tracking_height = compute_tracking_resize(frame_width, frame_height)
                tracking_to_display_scale = 1.0 / tracking_scale if tracking_scale > 0 else 1.0
                base_delay = max(1, int(1000 / fps)) if fps > 0 else 33
                delay = base_delay
                speed_multiplier = 1.0
                last_scale = frame_width / 1920
                tracker.reset()
                paused = True
                log_tracking_resize("Tracking input", frame_width, frame_height, tracking_width, tracking_height, tracking_scale)
                ret, frame = cap.read()
                if ret:
                    tracking_frame = resize_for_tracking(frame, tracking_width, tracking_height, tracking_scale)
                    result = tracker.process_frame(tracking_frame)
                    tracks = extract_tracks(result, tracking_to_display_scale)
                    last_track_count = len(tracks)
                    last_track_vis = draw_track_view(frame, tracks, last_scale, tracker_label)
        elif key == ord('s'):
            selection = config_dialog.select(
                default_video_path=video_path,
                default_mode=TRACKING_MODE_ULTRA,
                default_tracker=tracker_label,
            )
            if selection:
                next_selection = selection
                break
        elif key == ord('p'):
            paused = True
            params_dialog = TrackerParamsDialog(script_dir)
            params_result = params_dialog.select(tracker_label)
            if params_result:
                if params_result["apply_to_current"]:
                    tracker_yaml = params_result["yaml_path"]
                    tracker.set_tracker_yaml(tracker_yaml)
                    tracker.reset()
                    paused = True
                    ret, frame = cap.read()
                    if ret:
                        tracking_frame = resize_for_tracking(frame, tracking_width, tracking_height, tracking_scale)
                        result = tracker.process_frame(tracking_frame)
                        tracks = extract_tracks(result, tracking_to_display_scale)
                        last_track_count = len(tracks)
                        last_track_vis = draw_track_view(frame, tracks, last_scale, tracker_label)
                else:
                    print(
                        f"Saved custom params for {params_result['tracker']}. "
                        "Use Settings (S) to switch trackers."
                    )
        elif key == ord('+') or key == ord('='):
            delay = max(1, delay - 5)
            speed_multiplier = base_delay / delay
        elif key == ord('-'):
            delay = min(500, delay + 5)
            speed_multiplier = base_delay / delay

    cap.release()
    cv2.destroyAllWindows()
    return next_selection


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "yolo11hbb_base90+40_poc_w_full_poc_blur_gripper.pt")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("=" * 50)
    print("Bottle Tracking")
    print("=" * 50)

    selector = VideoSelector(script_dir)
    config_dialog = TrackingConfigDialog(script_dir)
    selection = config_dialog.select()

    if not selection:
        print("No video selected.")
        return

    while selection:
        video_path = selection["video_path"]
        tracking_mode = selection["mode"]
        tracker_choice = selection["tracker"]

        if tracking_mode == TRACKING_MODE_ULTRA:
            tracker_yaml = get_active_tracker_path(script_dir, tracker_choice)
            print(f"Tracking mode: {tracking_mode} ({tracker_choice})")
            selection = run_ultralytics_tracking(
                video_path, model_path, tracker_yaml, tracker_choice, selector, config_dialog, script_dir
            )
        else:
            print(f"Tracking mode: {tracking_mode}")
            selection = run_feature_tracking(video_path, model_path, selector, config_dialog, tracker_choice)


if __name__ == "__main__":
    main()
