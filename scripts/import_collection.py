"""Import saved research artifacts into an independent runtime, without source DB access.

Usage: python scripts/import_collection.py --project /path/to/research-project
Requires the optional `import` dependencies. Never opens or modifies the source DB.
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import storage as db
from retrieval.windows import KEYPOINTS


def import_collection(project):
    import numpy as np
    import pandas as pd

    project = Path(project).expanduser().resolve()
    with (project / 'data/pipeline/tracklet_manifest.csv').open() as f:
        manifest = list(csv.DictReader(f))
    exported = {}
    for p in (project / 'runtime/jobs').glob('*/exports/*/accepted.json'):
        for row in json.loads(p.read_text()).get('tracklets', []):
            exported[row['tracklet_id']] = row
    groups = {}
    for row in manifest:
        if (project / 'data/pipeline/poses' / f'{row["tracklet_id"]}.parquet').is_file():
            groups.setdefault(row['job_id'], []).append(row)
    result = []
    for jid, rows in groups.items():
        try:
            existing=db.get('jobs', jid)
            if not existing.get('imported_collection') or existing.get('import_complete'):
                result.append({'job_id': jid, 'skipped': 'already imported'})
                continue
        except KeyError:
            pass
        config_path = project / 'runtime/jobs' / jid / 'run_config.json'
        job = json.loads(config_path.read_text())
        job.update(imported_collection=True, imported_from='saved_tracklet_and_pose_artifacts',
                   imported_at=time.time(), source_status=job.get('status'), status='ready',
                   identity_status='ready', pose_status='ready', stage='Imported processed collection',
                   tracklet_count=len(rows), error=None, pause_requested=False, scores_stale=False)
        # Preserve the exact source path; never substitute a similarly long video.
        if job['match'] == 'inputs':
            job['match'] = 'Imported match · ' + jid[:8]
        for key in ['worker_pid', 'last_reset_event_id', 'export_path', 'export_status', 'identity_model']:
            job.pop(key, None)
        db.put('jobs', jid, job)
        count = 0
        for row in rows:
            tid = row['tracklet_id']
            try:
                db.get('tracklets', tid)
                count += 1
                continue
            except KeyError:
                pass
            table = pd.read_parquet(project / 'data/pipeline/poses' / f'{tid}.parquet')
            table = table.sort_values('original_match_time').drop_duplicates('original_match_time')
            pose_frames = []
            for record in table.to_dict('records'):
                kps = np.array([[record[f'kp_{k}_x'], record[f'kp_{k}_y'], record[f'kp_{k}_conf']] for k in KEYPOINTS], dtype=float)
                invalid = ~np.isfinite(kps).all(1)
                kps = np.nan_to_num(kps)
                kps[invalid] = 0
                pose_frames.append({'time': float(record['original_match_time']),
                                    'keypoints': kps.tolist(),
                                    'bbox': [float(record[f'bbox_{k}']) for k in ['x1', 'y1', 'x2', 'y2']],
                                    'iou': float(record['iou']), 'confidence': float(record['confidence'])})
            if not pose_frames:
                continue
            old_export = exported.get(tid)
            start, end = float(row['start_time']), float(row['end_time'])
            review_start = old_export['start_time'] if old_export else start
            review_end = old_export['end_time'] if old_export else end
            sid = tid.rsplit('-t', 1)[0]
            # Reuse original detector boxes if a saved export contains them. Otherwise
            # make the pose-derived box provenance explicit rather than inventing it.
            frames = old_export['bounding_boxes'] if old_export else [
                {'time': f['time'], 'frame': round(f['time'] * job['fps']), 'bbox': f['bbox'],
                 'confidence': f['confidence'], 'quality': 0, 'overlap': 0, 'sharpness': 0} for f in pose_frames]
            representative = None
            original_crops = sorted((project / 'runtime/jobs' / jid / 'crops').glob(f'{tid}-*.jpg'))
            if original_crops:
                original = original_crops[len(original_crops)//2]
                representative = f'jobs/{jid}/crops/{tid}-representative.jpg'
                target = db.STATE / representative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
            track = {'id': tid, 'job_id': jid, 'shot_id': sid, 'local_id': int(tid.rsplit('-t', 1)[1]),
                     'start': start, 'end': end, 'duration': end-start, 'frames': frames,
                     'crops': [], 'representative': representative, 'flags': ['imported_collection'],
                     'bounding_box_origin': 'detector' if old_export else 'pose_estimate',
                     'number_of_detections': len(frames), 'median_bbox_height': float(np.median([f['bbox'][3]-f['bbox'][1] for f in frames])),
                     'median_bbox_width': float(np.median([f['bbox'][2]-f['bbox'][0] for f in frames])),
                     'median_detection_confidence': 0, 'quality_score': 0}
            db.put('tracklets', tid, track, jid, sid)
            try:
                shot = db.get('shots', sid)
                shot.update(start=min(start, shot['start']), end=max(end, shot['end']))
            except KeyError:
                shot = {'id': sid, 'job_id': jid, 'index': int(sid.rsplit('-s', 1)[1]),
                        'start': start, 'end': end, 'accepted': True, 'status': 'complete',
                        'representatives': [], 'wide_shot_score': None,
                        'reasons': ['Imported processed collection; original shot boundary metadata unavailable']}
            db.put('shots', sid, shot, jid)
            pose = {'model': 'yolov8x-pose', 'tracklet_id': tid, 'source_hash': job['input_hash'],
                    'frames': pose_frames, 'frame_count': len(pose_frames),
                    'provenance': 'imported_research_cache_unverified_source_binding'}
            label = {'label': 1, 'status': 'accepted', 'start': review_start, 'end': review_end,
                     'crops': [], 'review_source': 'imported_unspecified'}
            score = {'score': old_export.get('match_score') if old_export else None,
                     'frames': old_export.get('frame_level_scores', []) if old_export else [],
                     'embedding_version': old_export.get('embedding_model_version') if old_export else None,
                     'flags': ['imported_collection'], 'band': 'uncertain'}
            with db.connect() as c:
                c.execute('INSERT OR REPLACE INTO pose_estimates VALUES (?,?,?)', (jid, tid, db.encode(pose)))
                c.execute('INSERT INTO labels(job,tracklet,kind,data,created) VALUES (?,?,?,?,?)',
                          (jid, tid, 'review', db.encode(label), time.time()))
                c.execute('INSERT OR REPLACE INTO scores VALUES (?,?,?)', (jid, tid, db.encode(score)))
            count += 1
        job = db.get('jobs', jid)
        job['import_complete'] = True
        job['completed_shots'] = len(db.listing('shots', jid))
        job['total_shots'] = job['completed_shots']
        job['accepted_shots'] = job['completed_shots']
        db.put('jobs', jid, job)
        result.append({'job_id': jid, 'tracklets': count, 'cached_poses': count,
                       'source_available': Path(job['path']).is_file()})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    args = parser.parse_args()
    print(json.dumps(import_collection(args.project), indent=2))
