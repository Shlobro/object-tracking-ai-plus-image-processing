import argparse
import glob
import os
import sys

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:
    print('ERROR: ultralytics is required. Install with: pip install ultralytics', file=sys.stderr)
    raise


def pick_video():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title='Select video file',
        filetypes=[('Video files', '*.mp4;*.avi;*.mov;*.mkv'), ('All files', '*.*')],
    )
    root.destroy()
    if not path:
        return None
    return path


def get_screen_size():
    try:
        import tkinter as tk
    except Exception:
        return (1920, 1080)
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return (int(w), int(h))


def get_window_size(win_name, fallback):
    try:
        x, y, w, h = cv2.getWindowImageRect(win_name)
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass
    return fallback


def letterbox_to_size(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def auto_weights():
    pts = glob.glob('*.pt')
    return pts[0] if pts else None


def best_bbox(results):
    # Axis-aligned bbox from boxes or OBB, choose highest confidence.
    if results.boxes is not None and len(results.boxes) > 0:
        confs = results.boxes.conf.detach().cpu().numpy()
        idx = int(np.argmax(confs))
        xyxy = results.boxes.xyxy[idx].detach().cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy.tolist()
        return (x1, y1, x2, y2, float(confs[idx]))

    obb = getattr(results, 'obb', None)
    if obb is None or len(obb) == 0:
        return None
    try:
        confs = obb.conf.detach().cpu().numpy()
        idx = int(np.argmax(confs))
    except Exception:
        confs = np.ones((len(obb),), dtype=np.float32)
        idx = 0

    # Try common OBB formats
    if hasattr(obb, 'xyxy') and obb.xyxy is not None:
        xyxy = obb.xyxy[idx].detach().cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy.tolist()
        return (x1, y1, x2, y2, float(confs[idx]))

    if hasattr(obb, 'xyxyxyxy') and obb.xyxyxyxy is not None:
        pts = obb.xyxyxyxy[idx].detach().cpu().numpy().reshape(-1, 2)
        x1 = int(np.min(pts[:, 0]))
        y1 = int(np.min(pts[:, 1]))
        x2 = int(np.max(pts[:, 0]))
        y2 = int(np.max(pts[:, 1]))
        return (x1, y1, x2, y2, float(confs[idx]))

    return None


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def build_mask(shape, center, radius, bbox=None):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = clamp(x1, 0, w - 1)
        x2 = clamp(x2, 0, w - 1)
        y1 = clamp(y1, 0, h - 1)
        y2 = clamp(y2, 0, h - 1)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)
    return mask


def detect_points(gray, center, radius, bbox, max_points):
    mask = build_mask(gray.shape, center, radius, bbox)
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=8,
        mask=mask,
        blockSize=7,
    )
    if pts is None:
        return None
    return pts.reshape(-1, 2)


def normalize_dirs(pts, center):
    dirs = []
    good_pts = []
    for p in pts:
        v = np.array([center[0] - p[0], center[1] - p[1]], dtype=np.float32)
        n = np.linalg.norm(v)
        if n < 1e-4:
            continue
        dirs.append(v / n)
        good_pts.append(p)
    if not dirs:
        return None, None
    return np.array(good_pts, dtype=np.float32), np.array(dirs, dtype=np.float32)


def estimate_center(points, dirs):
    # least squares intersection of lines
    if points is None or dirs is None or len(points) < 2:
        return None
    A = np.zeros((2, 2), dtype=np.float64)
    b = np.zeros((2,), dtype=np.float64)
    for p, d in zip(points, dirs):
        d = d / (np.linalg.norm(d) + 1e-9)
        I = np.eye(2)
        P = I - np.outer(d, d)
        A += P
        b += P @ p
    if np.linalg.cond(A) > 1e8:
        return None
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return (float(x[0]), float(x[1]))


def draw_line(img, p, d, color, length=2000, thickness=2):
    p1 = (int(p[0] - d[0] * length), int(p[1] - d[1] * length))
    p2 = (int(p[0] + d[0] * length), int(p[1] + d[1] * length))
    cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--radius', type=int, default=250)
    parser.add_argument('--min-points', type=int, default=12)
    parser.add_argument('--max-points', type=int, default=40)
    args = parser.parse_args()

    video = args.video or pick_video()
    if not video:
        print('No video selected.')
        return 1

    weights = args.weights or auto_weights()
    if not weights:
        print('No .pt weights found in current folder.')
        return 1

    model = YOLO(weights)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print('Failed to open video:', video)
        return 1

    yolo_enabled = True
    points = None
    dirs = None
    prev_gray = None
    last_bbox = None
    last_center = None
    yolo_frames = 0
    frame_idx = 0
    fps_alpha = 0.05
    fps_est = 0.0
    last_tick = cv2.getTickCount()
    did_resize = False
    paused = False
    last_combined_disp = None

    screen_w, screen_h = get_screen_size()
    win_name = 'YOLO (left) vs Points/Lines (right) - press q to quit'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win_name, 0, 0)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)

    while True:
        if paused:
            if last_combined_disp is not None:
                paused_disp = last_combined_disp.copy()
                cv2.rectangle(paused_disp, (10, 10), (430, 70), (0, 0, 0), -1)
                cv2.rectangle(paused_disp, (10, 10), (430, 70), (255, 255, 255), 2)
                cv2.putText(paused_disp, 'PAUSED - space to resume', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow(win_name, paused_disp)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                paused = False
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame_disp_left = frame.copy()
        frame_disp_right = frame.copy()
        frame_idx += 1
        # FPS estimate
        tick = cv2.getTickCount()
        dt = (tick - last_tick) / cv2.getTickFrequency()
        last_tick = tick
        if dt > 0:
            inst_fps = 1.0 / dt
            fps_est = inst_fps if fps_est == 0.0 else (fps_alpha * inst_fps + (1 - fps_alpha) * fps_est)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if yolo_enabled:
            results = model(frame, verbose=False)[0]
            bbox = best_bbox(results)
            if bbox is not None:
                x1, y1, x2, y2, conf = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                last_bbox = (x1, y1, x2, y2)
                last_center = (cx, cy)
                cv2.rectangle(frame_disp_left, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.circle(frame_disp_left, (int(cx), int(cy)), 7, (0, 0, 255), -1)
                cv2.putText(frame_disp_left, f'YOLO conf {conf:.2f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if points is None:
                    pts = detect_points(gray, (cx, cy), args.radius, (x1, y1, x2, y2), args.max_points)
                    if pts is not None and len(pts) >= args.min_points:
                        pts, d = normalize_dirs(pts, (cx, cy))
                        if pts is not None and len(pts) >= args.min_points:
                            points = pts
                            dirs = d
                yolo_frames += 1
            else:
                if last_bbox is not None:
                    x1, y1, x2, y2 = last_bbox
                    cv2.rectangle(frame_disp_left, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    if last_center is not None:
                        cv2.circle(frame_disp_left, (int(last_center[0]), int(last_center[1])), 5, (0, 0, 255), -1)
                    cv2.putText(frame_disp_left, 'YOLO: last bbox', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame_disp_left, 'YOLO: no detection', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                cv2.rectangle(frame_disp_left, (x1, y1), (x2, y2), (160, 160, 160), 3)
                if last_center is not None:
                    cv2.circle(frame_disp_left, (int(last_center[0]), int(last_center[1])), 5, (160, 160, 160), -1)
            cv2.putText(frame_disp_left, 'YOLO OFF', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

        if points is not None and prev_gray is not None:
            p0 = points.reshape(-1, 1, 2).astype(np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
            if p1 is not None:
                st = st.reshape(-1)
                p1 = p1.reshape(-1, 2)
                points = p1[st == 1]
                dirs = dirs[st == 1]
            if points is not None and len(points) < 2:
                # keep but cannot estimate
                pass

        # Draw points, lines, and estimated center on right view
        if points is not None and dirs is not None and len(points) > 0:
            for p, d in zip(points, dirs):
                cv2.circle(frame_disp_right, (int(p[0]), int(p[1])), 7, (255, 200, 0), -1)
                cv2.circle(frame_disp_right, (int(p[0]), int(p[1])), 10, (0, 0, 0), 2)
                draw_line(frame_disp_right, p, d, (255, 200, 0), length=2500, thickness=2)

            est = estimate_center(points, dirs)
            if est is not None:
                cv2.drawMarker(frame_disp_right, (int(est[0]), int(est[1])), (0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
                cv2.putText(frame_disp_right, 'EST', (int(est[0]) + 6, int(est[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame_disp_right, 'Estimate unavailable', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame_disp_right, f'Points: {len(points)}', (10, frame_disp_right.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        else:
            cv2.putText(frame_disp_right, 'Waiting for init points', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        prev_gray = gray

        combined = np.hstack([frame_disp_left, frame_disp_right])
        # Divider line between feeds
        mid_x = frame_disp_left.shape[1]
        cv2.line(combined, (mid_x, 0), (mid_x, combined.shape[0] - 1), (255, 255, 255), 2)

        # Scale down to fit screen with a small margin.
        max_w = max(320, screen_w - 80)
        max_h = max(240, screen_h - 120)
        scale = min(max_w / combined.shape[1], max_h / combined.shape[0], 1.0)
        if scale < 1.0:
            new_w = int(combined.shape[1] * scale)
            new_h = int(combined.shape[0] * scale)
            combined_disp = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            combined_disp = combined

        # UI stats (rendered in a header above the video)
        if yolo_enabled:
            status = 'INIT (YOLO ON)' if points is None else 'TRACK (YOLO ON)'
        else:
            status = 'TRACK (YOLO OFF)'
        pts_count = len(points) if points is not None else 0
        est = estimate_center(points, dirs) if points is not None and dirs is not None else None
        est_txt = f'{est[0]:.1f}, {est[1]:.1f}' if est is not None else 'N/A'
        err_txt = 'N/A'
        if yolo_enabled and last_center is not None and est is not None:
            dx = est[0] - last_center[0]
            dy = est[1] - last_center[1]
            err_txt = f'{(dx * dx + dy * dy) ** 0.5:.1f}px'
        info = [
            f'Frame: {frame_idx}',
            f'FPS: {fps_est:.1f}',
            f'Status: {status}',
            f'YOLO frames used: {yolo_frames}',
            f'Points: {pts_count}',
            f'Est center: {est_txt}',
            f'Est error vs YOLO: {err_txt}',
            'Keys: space=play/pause, q=quit',
        ]
        panel_w = max(720, int(combined_disp.shape[1] * 0.6))
        panel_h = 40 + 38 * len(info)
        header_h = panel_h + 20
        header = np.zeros((header_h, combined_disp.shape[1], 3), dtype=combined_disp.dtype)
        cv2.rectangle(header, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(header, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 2)
        for i, line in enumerate(info):
            cv2.putText(
                header,
                line,
                (24, 52 + i * 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.35,
                (255, 255, 255),
                3,
            )

        # Labels for the two panes (drawn on video area)
        cv2.putText(combined_disp, 'YOLO', (20, combined_disp.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(combined_disp, 'POINTS + LINES', (combined_disp.shape[1] // 2 + 20, combined_disp.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

        full_disp = np.vstack([header, combined_disp])
        # Fit to current window size without stretching (letterbox).
        win_w, win_h = get_window_size(win_name, (full_disp.shape[1], full_disp.shape[0]))
        if win_w > 0 and win_h > 0:
            display_img = letterbox_to_size(full_disp, win_w, win_h)
        else:
            display_img = full_disp

        cv2.imshow(win_name, display_img)
        # Ensure window has a visible size on first render only.
        if not did_resize and full_disp is not None:
            cv2.resizeWindow(win_name, full_disp.shape[1], full_disp.shape[0])
            did_resize = True
        last_combined_disp = full_disp
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            paused = True

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
