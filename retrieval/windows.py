"""Session-scoped pose windows. Never infer identity or cross a reviewed boundary."""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from . import storage as db

VERSION = 'pose-windows-v1.1'
CONFIG = {'duration': 2.0, 'stride': 1.0, 'sample_rate': 25, 'samples': 50,
          'confidence_threshold': .3, 'max_gap': .4}
KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
             'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
             'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
             'left_knee', 'right_knee', 'left_ankle', 'right_ankle']


def digest(value):
    return hashlib.sha256(db.encode(value).encode()).hexdigest()


def accepted(jobid):
    db.get('jobs', jobid)
    latest = {}
    for label in db.labels(jobid):
        if label['kind'] == 'review':
            latest[label['tracklet']] = label['data']
    return {tid: data for tid, data in latest.items() if data.get('status') == 'accepted'}


def review_source(label):
    if label.get('auto') or label.get('auto_accepted'):
        return 'automatic'
    return label.get('review_source', 'legacy_unspecified')


def signature(jobid):
    job = db.get('jobs', jobid)
    labels = accepted(jobid)
    with db.connect() as connection:
        poses = {r['tracklet']: hashlib.sha256(r['data'].encode()).hexdigest()
                 for r in connection.execute('SELECT tracklet,data FROM pose_estimates WHERE job=?', (jobid,))
                 if r['tracklet'] in labels}
    tracks = []
    for tid, label in sorted(labels.items()):
        track = db.get('tracklets', tid)
        tracks.append([tid, label, digest(track['frames']), poses.get(tid)])
    return digest([VERSION, CONFIG, job['input_hash'], job['target'], tracks])


def manifest_path(jobid):
    return db.jobdir(jobid) / 'windows' / 'manifest.json'


def read_manifest(jobid):
    db.get('jobs', jobid)
    p = manifest_path(jobid)
    return json.loads(p.read_text()) if p.exists() else None


def status(jobid):
    job = db.get('jobs', jobid)
    manifest = read_manifest(jobid)
    stale = bool(manifest and manifest['signature'] != signature(jobid))
    with db.connect() as c:
        pose_ids = {r[0] for r in c.execute('SELECT tracklet FROM pose_estimates WHERE job=?', (jobid,))}
    labels = accepted(jobid)
    return {'status': job.get('windows_status', 'idle'), 'error': job.get('windows_error'),
            'progress': job.get('windows_progress', 0), 'total': job.get('windows_total', 0),
            'accepted': len(labels), 'cached_poses': len(pose_ids & labels.keys()),
            'stale': stale, 'config': CONFIG, 'signature': manifest['signature'] if manifest else None,
            'count': len(manifest['windows']) if manifest else 0,
            'skipped': manifest.get('skipped', []) if manifest else [],
            'created': manifest.get('created') if manifest else None}


def current_manifest(jobid):
    manifest = read_manifest(jobid)
    if manifest is None:
        raise ValueError('Build windows first')
    if manifest['signature'] != signature(jobid):
        raise ValueError('Windows are stale after a review or pose change. Rebuild windows first.')
    return manifest


def window(jobid, windowid):
    manifest = current_manifest(jobid)
    row = next((w for w in manifest['windows'] if w['id'] == windowid), None)
    if row is None:
        raise KeyError(windowid)
    return row


def _normalise(points, confidence):
    """Hip-centred, torso-scaled COCO-17 skeleton; invalid values stay masked."""
    valid = confidence >= CONFIG['confidence_threshold']
    hips = valid[:, 11] & valid[:, 12]
    origin = (points[:, 11] + points[:, 12]) / 2
    lengths = np.stack([np.linalg.norm(points[:, 5] - points[:, 11], axis=1),
                        np.linalg.norm(points[:, 6] - points[:, 12], axis=1)], axis=1)
    torso_valid = np.stack([valid[:, 5] & valid[:, 11], valid[:, 6] & valid[:, 12]], axis=1)
    scale = (lengths * torso_valid).sum(1) / np.maximum(torso_valid.sum(1), 1)
    usable = hips & (scale >= 5) & (torso_valid.sum(1) > 0)
    mask = valid & usable[:, None]
    normalised = (points - origin[:, None, :]) / np.maximum(scale[:, None, None], 1)
    normalised[~mask] = 0
    return normalised.astype(np.float32), mask


def _sample(frames, start, end):
    """Resample only within one continuous interval, keeping source observations."""
    frames = sorted((r for r in frames if start - 1e-6 <= r['time'] < end - 1e-6), key=lambda r: r['time'])
    frames = list({float(r['time']): r for r in frames}.values())
    if len(frames) < 5:
        return None, 'Too few pose observations'
    times = np.array([r['time'] for r in frames], dtype=float)
    if times[0] - start > .12 or end - times[-1] > .12:
        return None, 'Incomplete pose coverage at window boundary'
    if np.max(np.diff(times)) > CONFIG['max_gap'] + 1e-6:
        return None, 'Pose gap exceeds 0.4 seconds'
    kps = np.array([r['keypoints'] for r in frames], dtype=float)
    boxes = np.array([r['bbox'] for r in frames], dtype=float)
    if kps.shape[1:] != (17, 3) or not np.isfinite(kps).all() or not np.isfinite(boxes).all():
        return None, 'Invalid pose coordinates'
    if np.mean((kps[:, 11:17, 2] >= .3).sum(1) >= 4) < .7:
        return None, 'Insufficient lower-body visibility'
    target = start + np.arange(CONFIG['samples']) / CONFIG['sample_rate']
    points = np.zeros((50, 17, 2), dtype=np.float32)
    confidence = np.zeros((50, 17), dtype=np.float32)
    for k in range(17):
        valid = kps[:, k, 2] >= .3
        if valid.sum() < 2:
            continue
        vt = times[valid]
        inside = (target >= vt[0] - .08) & (target <= vt[-1] + .08)
        right = np.clip(np.searchsorted(vt, target), 1, len(vt) - 1)
        inside &= (vt[right] - vt[right - 1] <= CONFIG['max_gap'] + 1e-6)
        for axis in range(2):
            points[:, k, axis] = np.interp(target, vt, kps[valid, k, axis])
        confidence[:, k] = np.interp(target, times, kps[:, k, 2])
        confidence[~inside, k] = 0
        points[~inside, k] = 0
    if np.mean((confidence[:, 11:17] >= .3).sum(1) >= 4) < .7:
        return None, 'Insufficient resampled lower-body visibility'
    normalised, mask = _normalise(points, confidence)
    if np.mean(mask[:, 11] & mask[:, 12]) < .7:
        return None, 'Insufficient hip or torso scale for normalisation'
    bbox = np.stack([np.interp(target, times, boxes[:, k]) for k in range(4)], axis=1)
    arrays = {'times': target, 'keypoints': points, 'kp_x': points[:, :, 0], 'kp_y': points[:, :, 1],
              'kp_conf': confidence, 'bbox': bbox, 'normalised': normalised,
              'valid_mask': mask, 'source_times': times,
              'observed_keypoints': kps, 'observed_bbox': boxes,
              'interpolated': np.min(abs(target[:, None] - times[None, :]), axis=1) > 1e-5}
    return arrays, None


def build(jobid, run_missing_pose=False):
    job = db.get('jobs', jobid)
    labels = accepted(jobid)
    if not labels:
        raise ValueError('Accept at least one tracklet before building windows')
    if job['status'] == 'processing' or job.get('identity_status') == 'scoring':
        raise ValueError('Finish or pause discovery and scoring before building windows')
    db.update_job(jobid, windows_status='working', windows_error=None, windows_progress=0,
                  windows_total=len(labels), windows_pause_requested=False)
    try:
        with db.connect() as c:
            poses = {r['tracklet']: json.loads(r['data']) for r in c.execute(
                'SELECT tracklet,data FROM pose_estimates WHERE job=?', (jobid,))}
        # Inference is opt-in; cached poses never require a model or source decode.
        for index, (tid, label) in enumerate(labels.items()):
            if db.get('jobs', jobid).get('windows_pause_requested'):
                db.update_job(jobid, windows_status='paused')
                return
            cached = poses.get(tid)
            needs_pose = cached is None or cached.get('coverage_start', -math.inf) > label['start'] + .001 or cached.get('coverage_end', math.inf) < label['end'] - .001
            if needs_pose and run_missing_pose and label['end'] - label['start'] >= 2:
                from pose.engine import estimate_tracklet
                track = db.get('tracklets', tid)
                track = track | {'start': label['start'], 'end': label['end'],
                                 'frames': [r for r in track['frames'] if label['start'] <= r['time'] < label['end']]}
                pose = estimate_tracklet(job, track)
                pose['source_hash'] = job['input_hash']
                pose['coverage_start'], pose['coverage_end'] = label['start'], label['end']
                poses[tid] = pose
                with db.connect() as c:
                    c.execute('INSERT OR REPLACE INTO pose_estimates VALUES (?,?,?)', (jobid, tid, db.encode(pose)))
            db.update_job(jobid, windows_progress=index + 1)
        if accepted(jobid) != labels:
            raise ValueError('Review decisions changed during pose preparation. Rebuild windows.')
        revision = signature(jobid)
        existing = read_manifest(jobid)
        if existing and existing['signature'] == revision and all((db.STATE / w['array_path']).exists() for w in existing['windows']):
            db.update_job(jobid, windows_status='ready')
            return existing
        directory = db.jobdir(jobid) / 'windows' / revision[:16]
        directory.mkdir(parents=True, exist_ok=True)
        windows, skipped = [], []
        for tid, label in labels.items():
            track = db.get('tracklets', tid)
            pose = poses.get(tid)
            start, end = float(label['start']), float(label['end'])
            reason = None
            if not track['start'] <= start < end <= track['end'] + .001:
                reason = 'Review interval is outside its tracklet'
            elif end - start < 2:
                reason = 'Reviewed interval is shorter than 2 seconds'
            elif pose is None:
                reason = 'No cached pose; enable missing-pose estimation to include this tracklet'
            elif pose.get('source_hash', job['input_hash']) != job['input_hash']:
                reason = 'Pose source does not match this video'
            else:
                try:
                    shot = db.get('shots', track['shot_id'])
                    start, end = max(start, shot['start']), min(end, shot['end'])
                except KeyError:
                    reason = 'Parent camera shot is missing'
            if reason:
                skipped.append({'tracklet_id': tid, 'start': start, 'end': end, 'reason': reason})
                continue
            for offset in range(max(0, math.floor((end - start - 2 + 1e-6) / CONFIG['stride']) + 1)):
                a, b = start + offset, start + offset + 2
                observations = [r for r in track['frames'] if a <= r['time'] < b]
                if len(observations) < 2 or observations[0]['time'] - a > .12 or b - observations[-1]['time'] > .12:
                    skipped.append({'tracklet_id': tid, 'start': a, 'end': b, 'reason': 'Incomplete tracking coverage at window boundary'})
                    continue
                if any(y['time'] - x['time'] > CONFIG['max_gap'] + 1e-6 for x, y in zip(observations, observations[1:])):
                    skipped.append({'tracklet_id': tid, 'start': a, 'end': b, 'reason': 'Tracking gap exceeds 0.4 seconds'})
                    continue
                arrays, reason = _sample(pose['frames'], a, b)
                if reason:
                    skipped.append({'tracklet_id': tid, 'start': a, 'end': b, 'reason': reason})
                    continue
                wid = f'{jobid}-{digest([tid, a, b, revision])[:16]}'
                path = directory / f'{wid}.npz'
                arrays['source_frame_indices'] = np.rint(arrays['source_times'] * job['fps']).astype(np.int64)
                np.savez_compressed(path, **arrays)
                windows.append({'id': wid, 'job_id': jobid, 'match': job['match'], 'target_player': job['target'],
                                'tracklet_id': tid, 'shot_id': track['shot_id'], 'start': a, 'end': b,
                                'duration': 2, 'source_fps': job['fps'], 'sample_rate': 25, 'samples': 50,
                                'source_observations': len(arrays['source_times']), 'review_source': review_source(label),
                                'pose_model': pose.get('model', 'unknown'), 'pose_provenance': pose.get('provenance', 'legacy_cache'),
                                'pose_quality': round(float((arrays['kp_conf'][:, 11:17] >= .3).mean()), 4),
                                'representative': track.get('representative'), 'flags': track.get('flags', []),
                                'array_path': str(path.relative_to(db.STATE))})
        # A concurrent review must never produce a falsely current manifest.
        if signature(jobid) != revision:
            raise ValueError('Review decisions changed during the build. Rebuild using the current intervals.')
        manifest = {'version': VERSION, 'signature': revision, 'created': time.time(), 'job_id': jobid,
                    'config': CONFIG, 'windows': windows, 'skipped': skipped}
        path = manifest_path(jobid)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(path)
        db.update_job(jobid, windows_status='ready')
        return manifest
    except Exception as error:
        db.update_job(jobid, windows_status='error', windows_error=str(error))
        raise


def preview(jobid, windowid):
    from .media import preview as track_preview
    row = window(jobid, windowid)
    return track_preview(jobid, row['tracklet_id'], row['start'], row['end'])
