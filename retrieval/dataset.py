"""Self-describing dataset exports from a frozen snapshot of review decisions."""
import csv
import io
import json
import math
import time
import uuid
import zipfile
from pathlib import Path

from . import storage as db
from . import windows


def selection(jobid, mode, window_ids=None, tracklet_ids=None):
    if mode not in {'tracklets', 'windows', 'both'}:
        raise ValueError('Choose tracklets, windows or both')
    labels = windows.accepted(jobid)
    if not labels:
        raise ValueError('Accept at least one tracklet before exporting')
    tids = list(labels) if tracklet_ids is None else list(dict.fromkeys(tracklet_ids))
    if any(t not in labels for t in tids):
        raise ValueError('Only currently accepted tracklets can be exported')
    rows = []
    manifest = None
    if mode in {'windows', 'both'}:
        manifest = windows.current_manifest(jobid)
        by_id = {w['id']: w for w in manifest['windows']}
        ids = list(by_id) if window_ids is None else list(dict.fromkeys(window_ids))
        if not ids or any(w not in by_id for w in ids):
            raise ValueError('Select at least one current window from this session')
        rows = [by_id[w] for w in ids]
    if mode == 'tracklets' and not tids:
        raise ValueError('Select at least one tracklet')
    return labels, tids, rows, manifest


def csv_text(rows):
    out = io.StringIO(newline='')
    if rows:
        fields = [k for k, v in rows[0].items() if not isinstance(v, (list, dict))]
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return out.getvalue()


README = '''# Touchline player dataset

manifest.json describes this export; tracklets.json/csv link reviewed portions to
source footage; windows.json/csv link each selected window to its parent tracklet.
No movement, health, injury or speed labels are inferred.

All start/end times are seconds relative to the original source video, with
half-open intervals [start, end). They are not the broadcast match clock.
Image coordinates use original source pixels: x right, y down; boxes are xyxy.
Tracklet bounding_box_origin identifies imported pose boxes where detector boxes
were unavailable. Enlarging crops does not add image detail.

Each windows/*.npz contains 50 samples at exactly 25 Hz across a 2-second window:
times = start + arange(50)/25 (last sample start+1.96). These are resampled pose
points, not 50 independently observed video frames. source_times,
source_frame_indices, observed_keypoints and observed_bbox preserve observations.
interpolated marks sample times absent from source_times. keypoints is (50,17,2),
kp_conf is (50,17), bbox is (50,4). COCO-17 order is listed in manifest.json.
normalised is hip-centred and torso-scaled; valid_mask marks usable points.
Invalid coordinates are zero, never treat zero as a valid observation without
checking confidence/valid_mask. No interpolation spans gaps over 0.4 seconds.

review_source is human, automatic, imported_unspecified or legacy_unspecified.
Retrieval scores are ranking scores, not calibrated identity probabilities.
An imported or automatically accepted sample has not necessarily been reviewed
by a human. Pose confidence is separate from identity confidence.

MP4 files are included only if requested. Player video uses source FPS; full-frame
context may extend outside the reviewed interval and is named separately.
The source video and model checkpoints are not bundled. Source footage and its
derivatives remain subject to the source dataset's permissions; this export
does not grant redistribution rights.
'''


def export(jobid, mode='both', window_ids=None, tracklet_ids=None, include_video=False, context=0):
    if not math.isfinite(context) or not 0 <= context <= 10:
        raise ValueError('Context must be between 0 and 10 seconds')
    before = windows.signature(jobid)
    labels, tids, rows, manifest = selection(jobid, mode, window_ids, tracklet_ids)
    job = db.get('jobs', jobid)
    if windows.signature(jobid) != before:
        raise ValueError('Review data changed while selecting the export. Try again.')
    if include_video and not Path(job['path']).is_file():
        raise ValueError('Source video is unavailable. Reconnect its drive to include video clips.')
    db.update_job(jobid, dataset_status='working', dataset_error=None, dataset_progress=0)
    directory = db.jobdir(jobid) / 'datasets' / uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    path = directory / 'dataset.zip'
    temp = directory / 'dataset.zip.partial'
    try:
        with db.connect() as c:
            scores = {r['tracklet']: json.loads(r['data']) for r in c.execute(
                'SELECT tracklet,data FROM scores WHERE job=?', (jobid,))}
        parent_ids = set(tids if mode in {'tracklets', 'both'} else []) | {w['tracklet_id'] for w in rows}
        tracks = {tid: db.get('tracklets', tid) for tid in parent_ids}
        track_rows = []
        for tid in sorted(parent_ids):
            t, label, score = tracks[tid], labels[tid], scores.get(tid, {})
            track_rows.append({'tracklet_id': tid, 'job_id': jobid, 'match': job['match'],
                               'target_player': job['target'], 'shot_id': t['shot_id'],
                               'start': label['start'], 'end': label['end'], 'duration': label['end'] - label['start'],
                               'source_fps': job['fps'], 'review_source': windows.review_source(label),
                               'identity_score': score.get('score'), 'embedding_model': score.get('embedding_version'),
                               'bounding_box_origin': t.get('bounding_box_origin', 'detector'),
                               'flags': t.get('flags', []), 'bounding_boxes': [r for r in t['frames'] if label['start'] <= r['time'] < label['end']],
                               'video': f'tracklets/{tid}.mp4' if include_video and mode in {'tracklets', 'both'} and tid in tids else None})
        window_rows = [{k: v for k, v in w.items() if k not in {'array_path', 'representative'}} |
                       {'array': f'windows/{w["id"]}.npz', 'video': f'windows/{w["id"]}.mp4' if include_video else None} for w in rows]
        for row in track_rows + window_rows:
            row['context_video'] = str(Path(row['video']).with_name(Path(row['video']).stem + '-context.mp4')) if row['video'] and context else None
            row['context_start'] = max(job['start'], row['start']-context) if row['context_video'] else None
            row['context_end'] = min(job['end'], row['end']+context) if row['context_video'] else None
        info = {'schema_version': 1, 'created': time.time(), 'mode': mode, 'job_id': jobid,
                'source': {'filename': job.get('filename', Path(job['path']).name), 'sha256': job['input_hash'],
                           'width': job['width'], 'height': job['height'], 'fps': job['fps']},
                'match': job['match'], 'target_player': job['target'], 'timestamp_basis': 'original_source_seconds',
                'intervals': '[start,end)', 'models': job.get('versions', {}),
                'settings': job['config'], 'window_config': windows.CONFIG,
                'keypoint_order': windows.KEYPOINTS, 'snapshot_signature': before,
                'window_build_signature': manifest['signature'] if manifest else None,
                'tracklet_count': len(track_rows), 'window_count': len(window_rows),
                'included_video': include_video, 'context_seconds': context}
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('README.md', README)
            archive.writestr('manifest.json', json.dumps(info, indent=2, allow_nan=False))
            archive.writestr('tracklets.json', db.encode(track_rows))
            archive.writestr('tracklets.csv', csv_text(track_rows))
            archive.writestr('windows.json', db.encode(window_rows))
            archive.writestr('windows.csv', csv_text(window_rows))
            if manifest:
                archive.writestr('skipped.json', db.encode(manifest['skipped']))
            for w in rows:
                archive.write(db.STATE / w['array_path'], f'windows/{w["id"]}.npz')
            if include_video:
                from .media import crop_clip, original_clip
                clips = [(tid, labels[tid]['start'], labels[tid]['end'], f'tracklets/{tid}.mp4')
                         for tid in tids if mode in {'tracklets', 'both'}]
                clips += [(w['tracklet_id'], w['start'], w['end'], f'windows/{w["id"]}.mp4') for w in rows]
                db.update_job(jobid, dataset_total=len(clips))
                for index, (tid, start, end, name) in enumerate(clips):
                    clip = directory / name
                    clip.parent.mkdir(parents=True, exist_ok=True)
                    crop_clip(job, tracks[tid], start, end, clip)
                    archive.write(clip, name)
                    if context:
                        a, b = max(job['start'], start-context), min(job['end'], end+context)
                        full = clip.with_name(clip.stem + '-context.mp4')
                        original_clip(job, a, b, full)
                        archive.write(full, str(Path(name).with_name(full.name)))
                    db.update_job(jobid, dataset_progress=index+1)
        if windows.signature(jobid) != before:
            raise ValueError('Review or pose data changed during export. Prepare a new dataset.')
        temp.replace(path)
        db.update_job(jobid, dataset_status='ready', dataset_path=str(path.relative_to(db.STATE)),
                      dataset_signature=before, dataset_mode=mode, dataset_window_ids=[w['id'] for w in rows],
                      dataset_has_windows=bool(rows), dataset_bytes=path.stat().st_size)
        return path
    except Exception as error:
        temp.unlink(missing_ok=True)
        db.update_job(jobid, dataset_status='error', dataset_error=str(error))
        raise


def ready_path(jobid):
    job = db.get('jobs', jobid)
    if job.get('dataset_status') != 'ready' or not job.get('dataset_path'):
        raise ValueError('Prepare a dataset first')
    if job.get('dataset_signature') != windows.signature(jobid):
        raise ValueError('Dataset is stale. Prepare a new export after your changes.')
    path = db.STATE / job['dataset_path']
    if not path.is_file():
        raise ValueError('Dataset file is missing. Prepare it again.')
    return path
