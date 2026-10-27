"""YOLOv8x-pose estimation on accepted tracklets.

Players in broadcast wide-shots are typically ~16-50px tall, far too small for
pose detection. Strategy: crop a padded region around the tracklet bbox, upscale
to ~384px tall, run pose on that, then map keypoints back to original coords.
"""
import json, time, os, subprocess
from pathlib import Path
import cv2, numpy as np

from retrieval import storage as db

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = db.MODELS / 'yolov8x-pose.pt'

# COCO skeleton limb connections (pairs of keypoint indices)
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # upper body
    (5, 11), (6, 12), (11, 12),                   # torso
    (11, 13), (13, 15), (12, 14), (14, 16),       # legs
    (0, 1), (0, 2), (1, 3), (2, 4),               # face
]
KP_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# Target height for the upscaled crop (pixels)
CROP_TARGET_H = 384

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO
    if not MODEL_PATH.exists():
        raise ValueError('Pose model is missing. Run python scripts/setup_models.py --pose first.')
    _model = YOLO(str(MODEL_PATH))
    return _model


def _iou(a, b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def _seek_frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def _crop_and_upscale(frame, bbox, target_h=CROP_TARGET_H, pad_factor=2.5):
    """Crop around bbox with generous padding, upscale so person is large enough for pose.

    Returns (crop, scale, ox, oy) where ox/oy is the crop origin in original coords.
    To map crop-space coords back to original: orig_x = crop_x/scale + ox
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # Pad generously so the model sees context around the player
    pad_w = bw * pad_factor
    pad_h = bh * pad_factor
    half_w = max(pad_w, bw * 3) / 2
    half_h = max(pad_h, bh * 3) / 2

    # Keep aspect roughly square-ish for better detection
    half_side = max(half_w, half_h)

    crop_x1 = max(0, int(cx - half_side))
    crop_y1 = max(0, int(cy - half_side))
    crop_x2 = min(fw, int(cx + half_side))
    crop_y2 = min(fh, int(cy + half_side))

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    ch, cw = crop.shape[:2]
    if ch < 10 or cw < 10:
        return None, 1.0, 0, 0

    # Scale so crop height = target_h
    scale = target_h / ch
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return upscaled, scale, crop_x1, crop_y1


def _draw_skeleton(frame, keypoints, color=(0, 255, 0), thickness=2, radius=4):
    """Draw keypoints and limbs on frame. keypoints: list of [x,y,conf]."""
    pts = [(int(kp[0]), int(kp[1])) if kp[2] > 0.3 else None for kp in keypoints]
    for i, pt in enumerate(pts):
        if pt is not None:
            cv2.circle(frame, pt, radius, color, -1)
    for a, b in SKELETON:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(frame, pts[a], pts[b], color, thickness)
    return frame


def estimate_tracklet(job, tracklet):
    """Run pose estimation on one tracklet. Returns frame-level keypoints."""
    video_path = job['path']
    if not Path(video_path).is_file():
        raise ValueError('Source video is unavailable. Reconnect its drive or relink the exact source file.')
    model = load_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video_path}')

    pose_dir = db.STATE / 'jobs' / tracklet['job_id'] / 'pose'
    pose_dir.mkdir(parents=True, exist_ok=True)

    results = []
    annotated_frames = []

    try:
        for fr in tracklet['frames']:
            frame = _seek_frame(cap, fr['time'])
            if frame is None:
                continue

            track_bbox = fr['bbox']  # [x1, y1, x2, y2]

            # Crop and upscale around the tracklet bbox
            crop, scale, ox, oy = _crop_and_upscale(frame, track_bbox)
            if crop is None:
                continue

            # Run pose on upscaled crop
            preds = model(crop, verbose=False)
            if len(preds) == 0 or preds[0].keypoints is None:
                continue

            boxes_crop = preds[0].boxes.xyxy.cpu().numpy()
            kps_crop = preds[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

            if len(boxes_crop) == 0:
                continue

            # Map track bbox into crop space for matching
            track_in_crop = [
                (track_bbox[0] - ox) * scale,
                (track_bbox[1] - oy) * scale,
                (track_bbox[2] - ox) * scale,
                (track_bbox[3] - oy) * scale,
            ]

            # Match detection to tracklet bbox by IoU in crop space
            best_idx, best_iou = -1, 0.0
            for i, box in enumerate(boxes_crop):
                iou = _iou(track_in_crop, box.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            # Also try centre-distance fallback for very small bboxes
            if best_iou < 0.1:
                tcx = (track_in_crop[0] + track_in_crop[2]) / 2
                tcy = (track_in_crop[1] + track_in_crop[3]) / 2
                best_dist = float('inf')
                for i, box in enumerate(boxes_crop):
                    dcx = (box[0] + box[2]) / 2
                    dcy = (box[1] + box[3]) / 2
                    d = ((dcx - tcx)**2 + (dcy - tcy)**2)**0.5
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
                # Accept if centre is within reasonable distance
                crop_h = crop.shape[0]
                if best_dist > crop_h * 0.4:
                    continue

            if best_idx < 0:
                continue

            # Extract keypoints in crop space
            kps_in_crop = kps_crop[best_idx].tolist()  # [[x,y,conf], ...] x 17

            # Map keypoints back to original frame coordinates
            kps_orig = []
            for kp in kps_in_crop:
                kps_orig.append([
                    kp[0] / scale + ox,
                    kp[1] / scale + oy,
                    kp[2],
                ])

            matched_box_crop = boxes_crop[best_idx].tolist()
            matched_box_orig = [
                matched_box_crop[0] / scale + ox,
                matched_box_crop[1] / scale + oy,
                matched_box_crop[2] / scale + ox,
                matched_box_crop[3] / scale + oy,
            ]

            results.append({
                'time': fr['time'],
                'keypoints': kps_orig,
                'bbox': matched_box_orig,
                'iou': round(best_iou, 3),
                'confidence': float(preds[0].boxes.conf[best_idx]),
            })

            # Draw skeleton on the cropped view (much more visible)
            ann_crop = crop.copy()
            _draw_skeleton(ann_crop, kps_in_crop, color=(0, 255, 0), thickness=3, radius=5)
            # Draw matched bbox on crop
            bx = [int(v) for v in matched_box_crop]
            cv2.rectangle(ann_crop, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
            # Draw tracklet bbox on crop for reference
            tbx = [int(v) for v in track_in_crop]
            cv2.rectangle(ann_crop, (tbx[0], tbx[1]), (tbx[2], tbx[3]), (255, 255, 0), 1)
            annotated_frames.append((fr['time'], ann_crop))

    finally:
        cap.release()

    # Write annotated video from crops
    if annotated_frames:
        # Normalize all frames to same size (use first frame's dimensions)
        target_h, target_w = annotated_frames[0][1].shape[:2]
        out_path = pose_dir / f'{tracklet["id"]}.mp4'
        fps = len(annotated_frames) / max(tracklet['end'] - tracklet['start'], 0.1)
        fps = min(max(fps, 1), 30)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (target_w, target_h))
        for _, fr in annotated_frames:
            h, w = fr.shape[:2]
            if h != target_h or w != target_w:
                fr = cv2.resize(fr, (target_w, target_h))
            writer.write(fr)
        writer.release()
        # Re-encode with H.264 for browser playback
        h264_path = pose_dir / f'{tracklet["id"]}_h264.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(out_path),'-c:v','libx264','-preset','fast','-crf','23','-an',str(h264_path)], check=True)
        if h264_path.exists() and h264_path.stat().st_size > 0:
            h264_path.rename(out_path)

    return {
        'model': 'yolov8x-pose',
        'tracklet_id': tracklet['id'],
        'frames': results,
        'frame_count': len(results),
        'created': time.time(),
    }


def estimate_job(jobid):
    """Run pose estimation on all accepted tracklets for a job."""
    db.update_job(jobid, pose_status='processing', pose_error=None)
    try:
        job = db.get('jobs', jobid)

        with db.connect() as c:
            accepted = [r[0] for r in c.execute(
                "SELECT tracklet FROM labels WHERE job=? AND active=1 AND kind='review' "
                "AND json_extract(data,'$.status')='accepted'", (jobid,))]
            already = {r[0] for r in c.execute(
                'SELECT tracklet FROM pose_estimates WHERE job=?', (jobid,))}

        total = len(accepted)
        done = len(already)

        for tid in accepted:
            if tid in already:
                continue

            current = db.get('jobs', jobid)
            if current.get('pose_pause_requested'):
                db.update_job(jobid, pose_status='paused', pose_pause_requested=False,
                              pose_progress=done, pose_total=total)
                return {'done': done, 'total': total, 'paused': True}

            tracklet = db.get('tracklets', tid)
            result = estimate_tracklet(job, tracklet)

            with db.connect() as c:
                c.execute('INSERT OR REPLACE INTO pose_estimates VALUES (?,?,?)',
                          (jobid, tid, db.encode(result)))
            done += 1
            if done % 5 == 0:
                db.update_job(jobid, pose_progress=done, pose_total=total)

        db.update_job(jobid, pose_status='ready', pose_progress=done, pose_total=done)
        return {'done': done, 'total': total}

    except Exception as e:
        db.update_job(jobid, pose_status='error', pose_error=str(e))
        raise
