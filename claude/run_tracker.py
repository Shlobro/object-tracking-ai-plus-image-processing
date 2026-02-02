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
from tkinter import filedialog


class FeatureBasedTracker:
    def __init__(self, model_path, init_frames=5, num_features=30, search_radius=350):
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.init_frames = init_frames
        self.num_features = num_features
        self.search_radius = search_radius

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
        self.reset()

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
        self.frames_since_yolo = 0

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

    def process_frame(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        is_init_phase = self.frame_count <= self.init_frames

        # Always run YOLO to get ground truth for comparison
        yolo_center, yolo_bbox, yolo_conf = self.detect_with_yolo(frame)

        if yolo_center is not None:
            self.last_yolo_center = yolo_center
            self.last_yolo_bbox = yolo_bbox
            self.frames_since_yolo = 0
        else:
            self.frames_since_yolo += 1

        if is_init_phase:
            # Initialization phase - use YOLO and build feature set
            if yolo_center is not None:
                self.last_center = np.array(yolo_center)
                self.last_bbox = yolo_bbox

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

            feature_center = yolo_center
        else:
            # Tracking phase - features only (no YOLO for position)
            if self.prev_gray is not None:
                self.track_points_optical_flow(self.prev_gray, gray)

            feature_center = self.estimate_center_from_intersections()

            # Try to add new features if we're running low
            if feature_center is not None and len(self.tracked_points) < self.num_features // 3:
                self.add_new_features(gray, feature_center)

        self.prev_gray = gray.copy()

        # Calculate error vs YOLO (for display purposes)
        if feature_center is not None and yolo_center is not None:
            error = np.sqrt((feature_center[0] - yolo_center[0])**2 +
                           (feature_center[1] - yolo_center[1])**2)
            self.error_history.append(error)

        return feature_center, yolo_center, yolo_bbox, yolo_conf, is_init_phase

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
            'is_init': self.frame_count <= self.init_frames
        }


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


def draw_feature_view(frame, tracker, feature_center, yolo_center, yolo_bbox, is_init_phase, scale):
    vis = frame.copy()
    pt_radius = max(6, int(10 * scale))
    line_thick = max(2, int(3 * scale))
    cross_size = max(20, int(35 * scale))

    # Draw search radius
    if feature_center is not None:
        fc = tuple(map(int, feature_center))
        cv2.circle(vis, fc, tracker.search_radius, (60, 60, 60), 1, cv2.LINE_AA)

    # Draw each tracked feature
    for i, p in enumerate(tracker.tracked_points):
        if not p['active']:
            continue

        pt = tuple(map(int, p['point']))

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
        end_pt_tuple = tuple(map(int, end_pt))
        cv2.arrowedLine(vis, pt, end_pt_tuple, (0, 255, 255), line_thick, cv2.LINE_AA, tipLength=0.06)

        # Draw point number
        cv2.putText(vis, str(i), (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw estimated center
    if feature_center is not None:
        fc = tuple(map(int, feature_center))
        cv2.line(vis, (fc[0] - cross_size, fc[1]), (fc[0] + cross_size, fc[1]), (255, 0, 255), line_thick + 1)
        cv2.line(vis, (fc[0], fc[1] - cross_size), (fc[0], fc[1] + cross_size), (255, 0, 255), line_thick + 1)
        cv2.circle(vis, fc, int(cross_size * 0.5), (255, 0, 255), line_thick + 1)
        cv2.circle(vis, fc, max(3, int(6 * scale)), (255, 0, 255), -1)

    # Draw YOLO center for comparison (when tracking)
    if yolo_center is not None and not is_init_phase:
        yc = tuple(map(int, yolo_center))
        cv2.circle(vis, yc, pt_radius, (0, 0, 255), line_thick)
        # Draw error line
        if feature_center is not None:
            fc = tuple(map(int, feature_center))
            cv2.line(vis, fc, yc, (0, 0, 255), 1, cv2.LINE_AA)

    return vis


def draw_yolo_view(frame, yolo_center, yolo_bbox, yolo_conf, scale):
    vis = frame.copy()
    line_thick = max(2, int(3 * scale))
    pt_radius = max(8, int(12 * scale))
    cross_size = max(20, int(30 * scale))

    if yolo_bbox is not None:
        x1, y1, x2, y2 = map(int, yolo_bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), line_thick + 1)

    if yolo_center is not None:
        yc = tuple(map(int, yolo_center))
        cv2.circle(vis, yc, pt_radius, (0, 0, 255), -1)
        cv2.circle(vis, yc, pt_radius + 2, (255, 255, 255), line_thick)
        cv2.line(vis, (yc[0] - cross_size, yc[1]), (yc[0] + cross_size, yc[1]), (0, 0, 255), line_thick)
        cv2.line(vis, (yc[0], yc[1] - cross_size), (yc[0], yc[1] + cross_size), (0, 0, 255), line_thick)

    return vis


def create_display(yolo_vis, feature_vis, tracker, yolo_conf, total_frames, is_paused, speed_multiplier, display_width, view_mode):
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
    else:
        video_combined = cv2.resize(feature_vis, (video_width, video_height))
        cv2.putText(video_combined, "FEATURES", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

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
    view_names = ["BOTH", "YOLO", "FEAT"]
    cv2.putText(bar, f"V:{view_names[view_mode]}", (770, 32), font, 0.6, (255, 255, 0), 1)

    # Progress bar
    prog_x = 870
    prog_w = display_width - prog_x - 10
    if prog_w > 50:
        progress = stats['frame'] / total_frames if total_frames > 0 else 0
        cv2.rectangle(bar, (prog_x, 18), (prog_x + prog_w, 28), (80, 80, 80), -1)
        cv2.rectangle(bar, (prog_x, 18), (prog_x + int(prog_w * progress), 28), (0, 255, 0), -1)

    # Controls bar
    ctrl_height = 25
    ctrl_bar = np.zeros((ctrl_height, display_width, 3), dtype=np.uint8)
    ctrl_bar[:] = (30, 30, 30)
    controls = "SPACE:Play/Pause  V:View  R:Reset  O:Open  +/-:Speed  Scroll:Size  Q:Quit"
    cv2.putText(ctrl_bar, controls, (10, 18), font, 0.45, (120, 120, 120), 1)

    result = np.vstack([bar, video_combined, ctrl_bar])
    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "yolo11hbb_base90+40_poc_w_full_poc_blur_gripper.pt")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("=" * 50)
    print("Feature-Based Bottle Tracking")
    print("=" * 50)

    selector = VideoSelector(script_dir)
    video_path = selector.select_video()

    if not video_path:
        print("No video selected.")
        return

    print(f"Video: {os.path.basename(video_path)}")

    # More features, larger search radius
    tracker = FeatureBasedTracker(model_path, init_frames=5, num_features=30, search_radius=350)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"FPS: {fps:.1f}, Frames: {total_frames}")
    print("\nSPACE:Play/Pause V:View R:Reset O:Open Q:Quit")

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
    last_yolo_conf = 0
    last_scale = frame_width / 1920

    # Read first frame
    ret, frame = cap.read()
    if ret:
        feature_center, yolo_center, yolo_bbox, yolo_conf, is_init = tracker.process_frame(frame)
        last_yolo_conf = yolo_conf
        last_yolo_vis = draw_yolo_view(frame, yolo_center, yolo_bbox, yolo_conf, last_scale)
        last_feature_vis = draw_feature_view(frame, tracker, feature_center, yolo_center, yolo_bbox, is_init, last_scale)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker.reset()
                continue

            feature_center, yolo_center, yolo_bbox, yolo_conf, is_init = tracker.process_frame(frame)
            last_yolo_conf = yolo_conf
            last_yolo_vis = draw_yolo_view(frame, yolo_center, yolo_bbox, yolo_conf, last_scale)
            last_feature_vis = draw_feature_view(frame, tracker, feature_center, yolo_center, yolo_bbox, is_init, last_scale)

        if last_yolo_vis is not None and last_feature_vis is not None:
            display = create_display(
                last_yolo_vis, last_feature_vis, tracker, last_yolo_conf,
                total_frames, paused, speed_multiplier, display_width, view_mode
            )
            cv2.imshow(window_name, display)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('v'):
            view_mode = (view_mode + 1) % 3
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker.reset()
            paused = True
            ret, frame = cap.read()
            if ret:
                feature_center, yolo_center, yolo_bbox, yolo_conf, is_init = tracker.process_frame(frame)
                last_yolo_conf = yolo_conf
                last_yolo_vis = draw_yolo_view(frame, yolo_center, yolo_bbox, yolo_conf, last_scale)
                last_feature_vis = draw_feature_view(frame, tracker, feature_center, yolo_center, yolo_bbox, is_init, last_scale)
        elif key == ord('o'):
            new_video = selector.select_video()
            if new_video:
                cap.release()
                video_path = new_video
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                base_delay = max(1, int(1000 / fps)) if fps > 0 else 33
                delay = base_delay
                speed_multiplier = 1.0
                last_scale = frame_width / 1920
                tracker.reset()
                paused = True
                ret, frame = cap.read()
                if ret:
                    feature_center, yolo_center, yolo_bbox, yolo_conf, is_init = tracker.process_frame(frame)
                    last_yolo_conf = yolo_conf
                    last_yolo_vis = draw_yolo_view(frame, yolo_center, yolo_bbox, yolo_conf, last_scale)
                    last_feature_vis = draw_feature_view(frame, tracker, feature_center, yolo_center, yolo_bbox, is_init, last_scale)
        elif key == ord('+') or key == ord('='):
            delay = max(1, delay - 5)
            speed_multiplier = base_delay / delay
        elif key == ord('-'):
            delay = min(500, delay + 5)
            speed_multiplier = base_delay / delay

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
