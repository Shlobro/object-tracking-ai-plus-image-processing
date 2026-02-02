"""
Feature-based bottle tracking system.
Uses YOLO for initial detection (first 5 frames), then tracks using
feature points and line intersections without AI.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math


class FeaturePoint:
    """Represents a trackable feature point with direction to object center."""
    def __init__(self, point, center, descriptor=None):
        self.point = np.array(point, dtype=np.float32)
        self.descriptor = descriptor
        # Vector from point to center
        self.direction = np.array(center) - np.array(point)
        self.distance = np.linalg.norm(self.direction)
        # Normalize direction
        if self.distance > 0:
            self.direction = self.direction / self.distance
        self.active = True
        self.lost_frames = 0

    def get_line_endpoint(self, length=500):
        """Get endpoint of line passing through point toward estimated center."""
        end = self.point + self.direction * length
        return end

    def estimate_center(self):
        """Estimate object center based on this point."""
        return self.point + self.direction * self.distance


class BottleTracker:
    def __init__(self, model_path, num_features=15, search_radius=250):
        """
        Initialize the tracker.

        Args:
            model_path: Path to YOLO .pt file
            num_features: Number of feature points to track (more = more robust)
            search_radius: Radius around detected object to find features
        """
        self.model = YOLO(model_path)
        self.num_features = num_features
        self.search_radius = search_radius

        # Feature detector - ORB is fast and robust
        self.orb = cv2.ORB_create(nfeatures=500)

        # For optical flow tracking
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.feature_points = []
        self.prev_gray = None
        self.frame_count = 0
        self.initialization_frames = 5
        self.initialized = False

        # For smoothing estimated position
        self.position_history = deque(maxlen=5)
        self.last_known_center = None

    def detect_bottle(self, frame):
        """Use YOLO to detect bottle and return bounding box center."""
        results = self.model(frame, verbose=False)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Get the first detection (assuming one bottle)
                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                return (center_x, center_y), (x1, y1, x2, y2)
        return None, None

    def find_good_features(self, gray, center, bbox):
        """
        Find good feature points around the detected object.
        Uses ORB keypoints that are within search_radius of center.
        """
        h, w = gray.shape
        cx, cy = center
        x1, y1, x2, y2 = bbox

        # Create mask for search area (ring around the object)
        mask = np.zeros(gray.shape, dtype=np.uint8)

        # Outer circle (search radius)
        cv2.circle(mask, (int(cx), int(cy)), self.search_radius, 255, -1)

        # Exclude the object itself (we want points around it, not on it)
        # Slightly expand bbox to avoid edge features
        margin = 20
        cv2.rectangle(mask,
                      (max(0, int(x1-margin)), max(0, int(y1-margin))),
                      (min(w, int(x2+margin)), min(h, int(y2+margin))),
                      0, -1)

        # Detect ORB keypoints
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)

        if keypoints is None or len(keypoints) < self.num_features:
            # Fallback: use Shi-Tomasi corners
            corners = cv2.goodFeaturesToTrack(gray, self.num_features * 2, 0.01, 10, mask=mask)
            if corners is not None:
                feature_points = []
                for corner in corners[:self.num_features]:
                    pt = corner.ravel()
                    fp = FeaturePoint(pt, center, descriptor=None)
                    feature_points.append(fp)
                return feature_points
            return []

        # Sort by response (quality) and take best ones
        sorted_kps = sorted(zip(keypoints, range(len(keypoints))),
                           key=lambda x: x[0].response, reverse=True)

        feature_points = []
        for kp, idx in sorted_kps[:self.num_features]:
            pt = kp.pt
            desc = descriptors[idx] if descriptors is not None else None
            fp = FeaturePoint(pt, center, descriptor=desc)
            feature_points.append(fp)

        return feature_points

    def track_features(self, prev_gray, curr_gray):
        """
        Track feature points from previous frame to current using optical flow.
        """
        if not self.feature_points:
            return

        # Prepare points for optical flow
        prev_pts = np.array([fp.point for fp in self.feature_points], dtype=np.float32)
        prev_pts = prev_pts.reshape(-1, 1, 2)

        # Calculate optical flow
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )

        # Update feature points
        active_features = []
        for i, (fp, st) in enumerate(zip(self.feature_points, status)):
            if st[0] == 1:  # Successfully tracked
                fp.point = next_pts[i].ravel()
                fp.lost_frames = 0
                active_features.append(fp)
            else:
                fp.lost_frames += 1
                if fp.lost_frames < 3:  # Keep for a few frames
                    active_features.append(fp)

        self.feature_points = active_features

    def estimate_center_from_features(self):
        """
        Estimate object center from feature points using line intersection.
        Uses RANSAC-like approach to handle outliers.
        """
        active_points = [fp for fp in self.feature_points if fp.active and fp.lost_frames == 0]

        if len(active_points) < 2:
            return self.last_known_center

        # Collect all center estimates
        estimates = []

        # Method 1: Direct estimates from each point
        for fp in active_points:
            est = fp.estimate_center()
            estimates.append(est)

        # Method 2: Line intersections (more robust)
        intersections = []
        for i in range(len(active_points)):
            for j in range(i + 1, len(active_points)):
                intersection = self.find_line_intersection(
                    active_points[i], active_points[j]
                )
                if intersection is not None:
                    intersections.append(intersection)

        if intersections:
            estimates.extend(intersections)

        if not estimates:
            return self.last_known_center

        # Use median for robustness against outliers
        estimates = np.array(estimates)
        center_x = np.median(estimates[:, 0])
        center_y = np.median(estimates[:, 1])

        estimated_center = (center_x, center_y)

        # Smooth with history
        self.position_history.append(estimated_center)
        if len(self.position_history) >= 2:
            avg_x = np.mean([p[0] for p in self.position_history])
            avg_y = np.mean([p[1] for p in self.position_history])
            estimated_center = (avg_x, avg_y)

        self.last_known_center = estimated_center
        return estimated_center

    def find_line_intersection(self, fp1, fp2):
        """
        Find intersection of two lines defined by feature points and their directions.
        """
        p1 = fp1.point
        d1 = fp1.direction
        p2 = fp2.point
        d2 = fp2.direction

        # Solve: p1 + t1*d1 = p2 + t2*d2
        # This gives us: t1*d1 - t2*d2 = p2 - p1

        # Check if lines are parallel
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-6:
            return None

        # Solve for t1
        dp = p2 - p1
        t1 = (dp[0] * d2[1] - dp[1] * d2[0]) / cross

        # Calculate intersection point
        intersection = p1 + t1 * d1

        # Validate: intersection should be roughly in the expected area
        # (within reasonable distance from the estimated centers)
        est1 = fp1.estimate_center()
        est2 = fp2.estimate_center()
        avg_est = (est1 + est2) / 2

        dist = np.linalg.norm(intersection - avg_est)
        if dist > 200:  # Too far from expected position
            return None

        return intersection

    def process_frame(self, frame):
        """
        Process a single frame.
        Returns: (estimated_center, yolo_center, is_using_yolo)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        yolo_center = None
        yolo_bbox = None
        estimated_center = None
        using_yolo = self.frame_count <= self.initialization_frames

        if using_yolo:
            # Initialization phase: use YOLO
            yolo_center, yolo_bbox = self.detect_bottle(frame)

            if yolo_center is not None:
                self.last_known_center = yolo_center

                # Find or update feature points
                new_features = self.find_good_features(gray, yolo_center, yolo_bbox)

                if self.frame_count == 1:
                    self.feature_points = new_features
                else:
                    # Track existing and merge with new
                    if self.prev_gray is not None:
                        self.track_features(self.prev_gray, gray)

                    # Add new features if we lost some
                    if len(self.feature_points) < self.num_features // 2:
                        self.feature_points = new_features

                estimated_center = yolo_center
        else:
            # Tracking phase: no YOLO, only features
            if self.prev_gray is not None:
                self.track_features(self.prev_gray, gray)

            estimated_center = self.estimate_center_from_features()

            # Still run YOLO for comparison (display only)
            yolo_center, yolo_bbox = self.detect_bottle(frame)

        self.prev_gray = gray.copy()

        return estimated_center, yolo_center, yolo_bbox, using_yolo

    def draw_visualization(self, frame, estimated_center, yolo_center, yolo_bbox, using_yolo):
        """
        Draw visualization on frame.
        """
        vis = frame.copy()

        # Draw feature points and their direction lines
        active_count = 0
        for fp in self.feature_points:
            if fp.lost_frames == 0:
                active_count += 1
                pt = tuple(map(int, fp.point))

                # Draw point
                cv2.circle(vis, pt, 5, (0, 255, 0), -1)
                cv2.circle(vis, pt, 7, (0, 200, 0), 2)

                # Draw direction line
                end_pt = fp.get_line_endpoint(150)
                end_pt = tuple(map(int, end_pt))
                cv2.line(vis, pt, end_pt, (0, 255, 255), 1)

        # Draw estimated center as a see-through crosshair
        if estimated_center is not None:
            est_pt = tuple(map(int, estimated_center))
            crosshair_size = 20
            # Draw crosshair lines (horizontal and vertical)
            cv2.line(vis, (est_pt[0] - crosshair_size, est_pt[1]),
                    (est_pt[0] + crosshair_size, est_pt[1]), (255, 0, 255), 2)
            cv2.line(vis, (est_pt[0], est_pt[1] - crosshair_size),
                    (est_pt[0], est_pt[1] + crosshair_size), (255, 0, 255), 2)
            # Small gap in center to see through - redraw center area with thinner line
            gap = 5
            cv2.line(vis, (est_pt[0] - gap, est_pt[1]),
                    (est_pt[0] + gap, est_pt[1]), (255, 0, 255), 1)
            cv2.line(vis, (est_pt[0], est_pt[1] - gap),
                    (est_pt[0], est_pt[1] + gap), (255, 0, 255), 1)
            cv2.putText(vis, "EST", (est_pt[0] + 20, est_pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Draw YOLO detection for reference
        if yolo_bbox is not None:
            x1, y1, x2, y2 = map(int, yolo_bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if yolo_center is not None:
            yolo_pt = tuple(map(int, yolo_center))
            cv2.circle(vis, yolo_pt, 8, (0, 0, 255), 2)

        # Status text
        status = "INITIALIZING (YOLO)" if using_yolo else "TRACKING (Features Only)"
        color = (0, 165, 255) if using_yolo else (0, 255, 0)
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Active Features: {active_count}/{len(self.feature_points)}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Calculate error if both centers available
        if estimated_center is not None and yolo_center is not None:
            error = np.sqrt((estimated_center[0] - yolo_center[0])**2 +
                          (estimated_center[1] - yolo_center[1])**2)
            cv2.putText(vis, f"Error: {error:.1f}px", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis


def create_side_by_side(yolo_frame, feature_frame):
    """Create side-by-side comparison view."""
    h1, w1 = yolo_frame.shape[:2]
    h2, w2 = feature_frame.shape[:2]

    # Resize if needed to match heights
    if h1 != h2:
        scale = h1 / h2
        feature_frame = cv2.resize(feature_frame, (int(w2 * scale), h1))

    # Add labels
    cv2.putText(yolo_frame, "YOLO Detection", (10, yolo_frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(feature_frame, "Feature Tracking", (10, feature_frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    combined = np.hstack([yolo_frame, feature_frame])
    return combined


def main():
    import sys

    # Paths
    model_path = "yolo11hbb_base90+40_poc_w_full_poc_blur_gripper.pt"
    video_path = "DJI_0829.MP4"  # Change to DJI_0830.MP4 if needed

    # Allow command line override
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    print(f"Loading model: {model_path}")
    print(f"Loading video: {video_path}")

    # Initialize tracker
    tracker = BottleTracker(model_path, num_features=15, search_radius=250)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}, Total frames: {total_frames}")
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  Q - Quit")
    print("  R - Reset tracker")
    print("  +/- - Speed up/slow down")

    paused = False
    delay = int(1000 / fps) if fps > 0 else 33

    # Create window
    cv2.namedWindow("Bottle Tracking Comparison", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                # Loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker = BottleTracker(model_path, num_features=15, search_radius=250)
                continue

            # Process frame
            estimated_center, yolo_center, yolo_bbox, using_yolo = tracker.process_frame(frame)

            # Create YOLO-only visualization
            yolo_frame = frame.copy()
            if yolo_bbox is not None:
                x1, y1, x2, y2 = map(int, yolo_bbox)
                cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if yolo_center is not None:
                cv2.circle(yolo_frame, tuple(map(int, yolo_center)), 10, (0, 0, 255), -1)

            status_yolo = "YOLO Active" if yolo_center else "YOLO: No Detection"
            cv2.putText(yolo_frame, status_yolo, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(yolo_frame, f"Frame: {tracker.frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Create feature tracking visualization
            feature_frame = tracker.draw_visualization(
                frame, estimated_center, yolo_center, yolo_bbox, using_yolo
            )

            # Combine side by side
            combined = create_side_by_side(yolo_frame, feature_frame)

            # Resize for display if too large
            max_width = 1920
            if combined.shape[1] > max_width:
                scale = max_width / combined.shape[1]
                combined = cv2.resize(combined, None, fx=scale, fy=scale)

            cv2.imshow("Bottle Tracking Comparison", combined)

        # Handle keyboard input
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker = BottleTracker(model_path, num_features=15, search_radius=250)
            print("Reset tracker")
        elif key == ord('+') or key == ord('='):
            delay = max(1, delay - 10)
            print(f"Speed: {delay}ms delay")
        elif key == ord('-'):
            delay = min(500, delay + 10)
            print(f"Speed: {delay}ms delay")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
