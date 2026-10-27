"""Manual activity moment annotations and activity-folder export."""
import csv, json, math, os, shutil, time, uuid
from pathlib import Path
from . import storage as db

EXPORT_ACTIVITIES = {'walking', 'jogging', 'sprinting'}
ALL_ACTIVITIES = EXPORT_ACTIVITIES | {'standing_other', 'mixed', 'unclear'}

# ---------------------------------------------------------------------------
# Moment CRUD
# ---------------------------------------------------------------------------

def create_moment(job_id: str, tracklet_id: str, start_time: float, end_time: float,
                  activity: str, identity_label_id: int | None = None,
                  target_player: str | None = None, half: int | None = None,
                  shot_id: str | None = None) -> dict:
    """Create a new activity moment for a confirmed tracklet interval."""
    if activity not in ALL_ACTIVITIES:
        raise ValueError(f'Activity must be one of: {", ".join(sorted(ALL_ACTIVITIES))}')
    if end_time <= start_time:
        raise ValueError('End time must be after start time')
    if end_time - start_time < 0.1:
        raise ValueError('Moment must be at least 0.1 seconds')

    t = db.get('tracklets', tracklet_id)
    if t['job_id'] != job_id:
        raise ValueError('Tracklet belongs to another job')

    # Verify identity is accepted for this tracklet
    accepted = False
    with db.connect() as c:
        labels = c.execute(
            "SELECT data FROM labels WHERE job=? AND tracklet=? AND kind='review' AND active=1 ORDER BY id DESC LIMIT 1",
            (job_id, tracklet_id)).fetchone()
        if labels:
            ld = json.loads(labels[0])
            if ld.get('status') == 'accepted':
                accepted = True
    if not accepted:
        raise ValueError('Identity must be accepted before labelling activity')

    moment_id = uuid.uuid4().hex[:16]
    now = time.time()
    quality_flags = []
    duration = end_time - start_time
    if duration < 2.0:
        quality_flags.append('short_duration')
    if duration > 8.0:
        quality_flags.append('long_duration')

    moment = {
        'id': moment_id,
        'job_id': job_id,
        'tracklet_id': tracklet_id,
        'target_player': target_player or db.get('jobs', job_id).get('target', 'Unknown'),
        'half': half,
        'start_time': start_time,
        'end_time': end_time,
        'activity': activity,
        'status': 'confirmed',
        'identity_label_id': identity_label_id,
        'shot_id': shot_id or t.get('shot_id'),
        'quality_flags': quality_flags,
        'revision': 1,
        'created': now,
        'updated': now,
    }

    with db.connect() as c:
        # Check for overlapping moments on same tracklet
        overlaps = c.execute(
            '''SELECT id FROM activity_moments
               WHERE job_id=? AND tracklet_id=? AND status='confirmed'
               AND start_time < ? AND end_time > ?''',
            (job_id, tracklet_id, end_time, start_time)).fetchall()
        if overlaps:
            raise ValueError('Moment overlaps with existing confirmed moment')

        c.execute('''INSERT INTO activity_moments
                     (id, job_id, tracklet_id, target_player, half,
                      start_time, end_time, activity, status,
                      identity_label_id, shot_id, quality_flags, data,
                      revision, created, updated)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                  (moment_id, job_id, tracklet_id, moment['target_player'],
                   half, start_time, end_time, activity, 'confirmed',
                   identity_label_id, moment['shot_id'],
                   json.dumps(quality_flags), db.encode({}),
                   1, now, now))

    db.event(job_id, 'activity_moment_created', {
        'moment_id': moment_id, 'tracklet_id': tracklet_id,
        'activity': activity, 'start': start_time, 'end': end_time,
    })
    return moment

def update_moment(moment_id: str, **changes) -> dict:
    """Update an activity moment (relabel, trim, etc)."""
    with db.connect() as c:
        row = c.execute('SELECT * FROM activity_moments WHERE id=?', (moment_id,)).fetchone()
        if not row:
            raise KeyError(moment_id)
        m = dict(row)

        updates = {}
        if 'activity' in changes:
            if changes['activity'] not in ALL_ACTIVITIES:
                raise ValueError(f'Activity must be one of: {", ".join(sorted(ALL_ACTIVITIES))}')
            updates['activity'] = changes['activity']
        if 'start_time' in changes:
            updates['start_time'] = float(changes['start_time'])
        if 'end_time' in changes:
            updates['end_time'] = float(changes['end_time'])
        if 'status' in changes:
            updates['status'] = changes['status']

        if updates.get('end_time', m['end_time']) <= updates.get('start_time', m['start_time']):
            raise ValueError('End time must be after start time')

        updates['revision'] = m['revision'] + 1
        updates['updated'] = time.time()

        sets = ', '.join(f'{k}=?' for k in updates)
        vals = list(updates.values()) + [moment_id]
        c.execute(f'UPDATE activity_moments SET {sets} WHERE id=?', vals)

    db.event(m['job_id'], 'activity_moment_updated', {
        'moment_id': moment_id, 'changes': updates,
    })
    return {**m, **updates}

def delete_moment(moment_id: str):
    """Soft-delete a moment by marking it invalidated."""
    return update_moment(moment_id, status='invalidated')

def split_moment(moment_id: str, split_time: float) -> list[dict]:
    """Split a moment at a given time into two moments. Original is invalidated."""
    with db.connect() as c:
        row = c.execute('SELECT * FROM activity_moments WHERE id=?', (moment_id,)).fetchone()
        if not row:
            raise KeyError(moment_id)
        m = dict(row)

    if split_time <= m['start_time'] or split_time >= m['end_time']:
        raise ValueError('Split time must be within moment bounds')

    # Invalidate original
    update_moment(moment_id, status='invalidated')

    # Create two new moments — first keeps the original activity,
    # second gets 'unclear' as default until the user labels it
    m1 = create_moment(
        m['job_id'], m['tracklet_id'], m['start_time'], split_time,
        m['activity'], m['identity_label_id'], m['target_player'],
        m['half'], m['shot_id'],
    )
    m2 = create_moment(
        m['job_id'], m['tracklet_id'], split_time, m['end_time'],
        'unclear', m['identity_label_id'], m['target_player'],
        m['half'], m['shot_id'],
    )

    db.event(m['job_id'], 'activity_moment_split', {
        'original_id': moment_id, 'new_ids': [m1['id'], m2['id']],
        'split_time': split_time,
    })
    return [m1, m2]

def list_moments(job_id: str, tracklet_id: str | None = None,
                 activity: str | None = None,
                 include_invalidated: bool = False) -> list[dict]:
    """List activity moments for a job."""
    query = 'SELECT * FROM activity_moments WHERE job_id=?'
    params: list = [job_id]
    if tracklet_id:
        query += ' AND tracklet_id=?'
        params.append(tracklet_id)
    if activity:
        query += ' AND activity=?'
        params.append(activity)
    if not include_invalidated:
        query += " AND status='confirmed'"
    query += ' ORDER BY start_time'

    with db.connect() as c:
        rows = c.execute(query, params).fetchall()
    return [dict(r) | {
        'quality_flags': json.loads(r['quality_flags'] or '[]'),
        'data': json.loads(r['data'] or '{}'),
    } for r in rows]

def undo_last_moment(job_id: str) -> dict:
    """Undo the last activity moment action (delete or restore)."""
    with db.connect() as c:
        row = c.execute(
            'SELECT * FROM activity_moments WHERE job_id=? ORDER BY updated DESC LIMIT 1',
            (job_id,)).fetchone()
        if not row:
            raise ValueError('No activity moments to undo')
        m = dict(row)

    if m['status'] == 'confirmed':
        return update_moment(m['id'], status='invalidated')
    elif m['status'] == 'invalidated':
        return update_moment(m['id'], status='confirmed')
    return m

def invalidate_moments_for_tracklet(job_id: str, tracklet_id: str):
    """Invalidate all moments when identity is rejected/reset.
    Pass '__all__' to invalidate every moment for a job (used by start-over)."""
    with db.connect() as c:
        if tracklet_id == '__all__':
            c.execute('''UPDATE activity_moments SET status='identity_invalidated', updated=?
                         WHERE job_id=? AND status='confirmed' ''',
                      (time.time(), job_id))
        else:
            c.execute('''UPDATE activity_moments SET status='identity_invalidated', updated=?
                         WHERE job_id=? AND tracklet_id=? AND status='confirmed' ''',
                      (time.time(), job_id, tracklet_id))
    db.event(job_id, 'moments_identity_invalidated', {'tracklet_id': tracklet_id})

# ---------------------------------------------------------------------------
# Activity-folder export
# ---------------------------------------------------------------------------

def export_activity_collection(job_id: str, match_id: str | None = None,
                               target_player_id: str | None = None) -> str:
    """Export confirmed activity moments into walking/jogging/sprinting folders."""
    from .media import crop_clip

    job = db.get('jobs', job_id)
    moments = list_moments(job_id)

    # Only export confirmed moments with exportable activities
    exportable = [m for m in moments if m['activity'] in EXPORT_ACTIVITIES]
    if not exportable:
        raise ValueError('No confirmed walking/jogging/sprinting moments to export')

    # Build export directory
    safe_match = match_id or job.get('match', 'unknown_match')
    safe_match = safe_match.replace('/', '_').replace('\\', '_')[:80]
    safe_player = target_player_id or job.get('target', 'unknown_player')
    safe_player = safe_player.replace('/', '_').replace('\\', '_')[:40]

    export_root = db.jobdir(job_id) / 'exports' / 'activity'
    export_dir = export_root / safe_match / safe_player

    # Create activity folders
    for act in EXPORT_ACTIVITIES:
        (export_dir / act).mkdir(parents=True, exist_ok=True)

    manifest_items = []
    snapshot_id = uuid.uuid4().hex[:12]

    for m in exportable:
        t = db.get('tracklets', m['tracklet_id'])
        activity = m['activity']
        half_label = f'half{m["half"]}' if m.get('half') else 'half1'

        # Filename: moment_id__half__start__end.mp4
        start_str = f'{m["start_time"]:.2f}'.replace('.', 'p')
        end_str = f'{m["end_time"]:.2f}'.replace('.', 'p')
        filename = f'{m["id"]}__{half_label}__{start_str}__{end_str}.mp4'
        clip_path = export_dir / activity / filename

        # Render crop clip at source fps
        crop_clip(job, t, m['start_time'], m['end_time'], clip_path)

        # Build manifest entry
        entry = {
            'moment_id': m['id'],
            'target_player': m['target_player'],
            'match': job.get('match', ''),
            'half': m.get('half'),
            'source_path': job.get('path', ''),
            'source_hash': job.get('input_hash', ''),
            'job_id': job_id,
            'shot_id': m.get('shot_id', ''),
            'tracklet_id': m['tracklet_id'],
            'start_time': m['start_time'],
            'end_time': m['end_time'],
            'duration': m['end_time'] - m['start_time'],
            'activity': activity,
            'review_status': m['status'],
            'quality_flags': m.get('quality_flags', []),
            'revision': m['revision'],
            'clip_filename': filename,
            'clip_folder': activity,
            'source_fps': job.get('fps', 25.0),
            'export_fps': job.get('fps', 25.0),
            'crop_policy': {'padding': 0.125, 'canvas': '256x512', 'interpolation': 'linear'},
            'bounding_boxes': [f for f in t.get('frames', [])
                               if m['start_time'] <= f.get('time', 0) < m['end_time']],
        }
        manifest_items.append(entry)

    # Write CSV manifest
    csv_fields = [
        'moment_id', 'target_player', 'match', 'half', 'job_id', 'shot_id',
        'tracklet_id', 'start_time', 'end_time', 'duration', 'activity',
        'review_status', 'revision', 'clip_filename', 'clip_folder',
        'source_fps', 'export_fps',
    ]
    with open(export_dir / 'moments.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(manifest_items)

    # Write JSON manifest
    json_manifest = {
        'export_id': snapshot_id,
        'job_id': job_id,
        'match': job.get('match', ''),
        'target_player': job.get('target', ''),
        'source_path': job.get('path', ''),
        'source_hash': job.get('input_hash', ''),
        'timestamp_basis': 'seconds relative to original source video half',
        'crop_policy': {'padding': 0.125, 'canvas_width': 256, 'canvas_height': 512,
                        'interpolation': 'linear_between_detections'},
        'settings': job.get('config', {}),
        'versions': job.get('versions', {}),
        'counts': {
            'walking': sum(1 for m in manifest_items if m['activity'] == 'walking'),
            'jogging': sum(1 for m in manifest_items if m['activity'] == 'jogging'),
            'sprinting': sum(1 for m in manifest_items if m['activity'] == 'sprinting'),
            'total': len(manifest_items),
        },
        'moments': manifest_items,
    }
    (export_dir / 'moments.json').write_text(json.dumps(json_manifest, indent=2))

    # Save export snapshot
    with db.connect() as c:
        c.execute('''INSERT INTO export_snapshots
                     (id, job_id, match_id, target_player, moment_ids,
                      crop_policy, output_path, data, created)
                     VALUES (?,?,?,?,?,?,?,?,?)''',
                  (snapshot_id, job_id, match_id, job.get('target', ''),
                   json.dumps([m['moment_id'] for m in manifest_items]),
                   db.encode(json_manifest['crop_policy']),
                   str(export_dir), db.encode(json_manifest['counts']),
                   time.time()))

    db.event(job_id, 'activity_export', {
        'snapshot_id': snapshot_id,
        'counts': json_manifest['counts'],
        'path': str(export_dir),
    })

    return str(export_dir.relative_to(db.STATE))
