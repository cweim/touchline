"""Tests for M1A–M1D: catalogue, downloads, generic player, activity moments, export."""
import csv, json, time
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2, numpy as np, pytest
from fastapi.testclient import TestClient
from retrieval import storage as db

@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'STATE', tmp_path)
    monkeypatch.setattr(db, 'DB', tmp_path / 'test.sqlite')
    db.init()
    from retrieval.catalog import migrate
    migrate()
    return tmp_path

def make_video(state, name='source.mp4'):
    source = state / name
    v = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 25, (320, 180))
    for i in range(300):
        img = np.full((180, 320, 3), (30, 120, 30), np.uint8)
        for k in range(8):
            cv2.rectangle(img, (k*36+4, 60), (k*36+23, 99), (230, 230, 230) if k%2 else (30, 30, 150), -1)
        v.write(img)
    v.release()
    return source

def seed_job(state, target='Marcelo', job_id='job', cutoff=None):
    source = make_video(state)
    config = {'reference_cutoff': cutoff, 'embedding': 'osnet', 'likely_threshold': .8, 'review_threshold': .55}
    j = {'id': job_id, 'path': str(source), 'target': target, 'filename': 'source.mp4',
         'match': 'Test Match', 'input_hash': db.sha_file(source), 'width': 320, 'height': 180,
         'fps': 25, 'duration': 12, 'start': 0, 'end': 12, 'status': 'ready', 'stage': 'Ready',
         'elapsed': 1, 'config': config, 'versions': {}, 'created': time.time(), 'sample_step': 2}
    db.put('jobs', job_id, j)
    # Create tracklets
    for i in range(4):
        t = i + .2
        frames = [{'frame': round((t + n*.08) * 25), 'time': t + n*.08,
                    'bbox': [40, 40, 65, 135], 'confidence': .9, 'quality': .9,
                    'overlap': 0, 'sharpness': 100} for n in range(10)]
        paths = []
        for k, row in enumerate(frames[::3]):
            rel = f'crop{job_id}-{i}-{k}.jpg'
            img = np.full((100, 40, 3), 200, np.uint8)
            cv2.imwrite(str(state / rel), img)
            paths.append({'path': rel, 'time': row['time'], 'bbox': row['bbox'],
                          'quality': .9, 'confidence': .9, 'overlap': 0, 'sharpness': 100})
        tr = {'id': f'{job_id}-t{i}', 'job_id': job_id, 'shot_id': f'{job_id}-s{i}',
              'local_id': 1, 'start': t, 'end': t + .8, 'duration': .8, 'frames': frames,
              'crops': paths, 'flags': [], 'number_of_detections': 10,
              'median_bbox_height': 95, 'median_bbox_width': 25,
              'median_detection_confidence': .9, 'quality_score': .9,
              'representative': paths[0]['path']}
        db.put('tracklets', tr['id'], tr, job_id, tr['shot_id'])
    return j

# ---------------------------------------------------------------------------
# M1A — Catalogue tests
# ---------------------------------------------------------------------------

def test_catalog_schema_migration_is_idempotent(state):
    """Repeated migrate calls must not duplicate tables or fail."""
    from retrieval.catalog import migrate
    migrate()
    migrate()
    with db.connect() as c:
        tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert 'matches' in tables
    assert 'source_assets' in tables
    assert 'local_assets' in tables
    assert 'download_tasks' in tables
    assert 'activity_moments' in tables

def test_catalog_match_id_is_stable():
    from retrieval.catalog import _match_id
    a = _match_id('src', 'path/to/match')
    b = _match_id('src', 'path/to/match')
    assert a == b
    c = _match_id('src', 'path/to/other')
    assert a != c

def test_catalog_parse_match_dir():
    from retrieval.catalog import _parse_match_dir
    m = _parse_match_dir('2015-11-03 - 22-45 Real Madrid 1 - 0 Paris SG')
    assert m['match_date'] == '2015-11-03'
    assert m['home_team'] == 'Real Madrid'
    assert m['away_team'] == 'Paris SG'
    # Non-matching name
    m2 = _parse_match_dir('something_else')
    assert 'raw_name' in m2
    assert 'match_date' not in m2

def test_catalog_upsert_no_duplicates(state):
    """Syncing twice must not create duplicate matches."""
    from retrieval.catalog import _match_id, _asset_id
    mid = _match_id('test_src', 'comp/2015/match')
    aid = _asset_id(mid, 'comp/2015/match/1_720p.mkv')
    with db.connect() as c:
        for _ in range(3):  # Insert 3 times
            c.execute('''INSERT INTO matches (id, source_id, remote_path, raw_name,
                         competition, season, match_date, home_team, away_team, data)
                         VALUES (?,?,?,?,?,?,?,?,?,?)
                         ON CONFLICT(source_id, remote_path) DO UPDATE SET
                         raw_name=excluded.raw_name''',
                      (mid, 'test_src', 'comp/2015/match', 'Test Match',
                       'UCL', '2015-2016', '2015-11-03', 'Team A', 'Team B', '{}'))
            c.execute('''INSERT INTO source_assets
                         (id, match_id, half, representation, remote_path,
                          revision, reported_size, asset_type, data)
                         VALUES (?,?,?,?,?,?,?,?,?)
                         ON CONFLICT(match_id, remote_path) DO UPDATE SET
                         reported_size=excluded.reported_size''',
                      (aid, mid, 1, '720p', 'comp/2015/match/1_720p.mkv',
                       'videos-720p', 5000000000, 'video', '{}'))
    with db.connect() as c:
        assert c.execute('SELECT COUNT(*) FROM matches').fetchone()[0] == 1
        assert c.execute('SELECT COUNT(*) FROM source_assets').fetchone()[0] == 1

def test_catalog_list_matches_filters(state):
    from retrieval.catalog import list_matches, _match_id
    # Insert test matches
    with db.connect() as c:
        for team, season in [('Real Madrid', '2015-2016'), ('Barcelona', '2016-2017')]:
            mid = _match_id('s', f'{team}/{season}')
            c.execute('''INSERT INTO matches (id, source_id, remote_path, raw_name,
                         home_team, away_team, season, competition, data)
                         VALUES (?,?,?,?,?,?,?,?,?)''',
                      (mid, 's', f'{team}/{season}', f'{team} match',
                       team, 'Opponent', season, 'La Liga', '{}'))
    all_m = list_matches()
    assert len(all_m) == 2
    filtered = list_matches(team='Real Madrid')
    assert len(filtered) == 1
    assert filtered[0]['home_team'] == 'Real Madrid'
    season_f = list_matches(season='2016-2017')
    assert len(season_f) == 1

def test_catalog_api_routes(state):
    from retrieval.api import app
    with TestClient(app) as c:
        assert c.get('/api/catalog/auth').status_code == 200
        assert c.get('/api/catalog/matches').status_code == 200
        assert c.get('/api/catalog/competitions').status_code == 200
        assert c.get('/api/catalog/seasons').status_code == 200
        assert c.get('/api/catalog/matches/nonexistent').status_code == 404

# ---------------------------------------------------------------------------
# M1B — Download tests
# ---------------------------------------------------------------------------

def test_download_queue_deduplication(state):
    from retrieval.acquisition import queue_download, list_downloads, cancel_download
    r1 = queue_download('asset1', 'match1', 'path/1_720p.mkv', reported_size=5000000)
    assert r1['state'] == 'queued'
    # Queue same asset again — should detect existing
    r2 = queue_download('asset1', 'match1', 'path/1_720p.mkv')
    assert r2['state'] == 'queued'
    assert r2['message'] == 'Download already queued'
    downloads = list_downloads()
    assert len(downloads) == 1
    cancel_download(r1['id'])
    downloads = list_downloads(state='cancelled')
    assert len(downloads) == 1

def test_download_retry_and_states(state):
    from retrieval.acquisition import queue_download, cancel_download, retry_download, list_downloads
    r = queue_download('a2', 'm2', 'path/2.mkv')
    cancel_download(r['id'])
    assert list_downloads(state='cancelled')[0]['id'] == r['id']
    retry_download(r['id'])
    assert list_downloads(state='queued')[0]['id'] == r['id']

def test_download_api_routes(state):
    from retrieval.api import app
    with TestClient(app) as c:
        assert c.get('/api/downloads').status_code == 200
        assert c.get('/api/downloads/disk-space').status_code == 200
        r = c.post('/api/downloads', json={
            'source_asset_id': 'test_asset', 'match_id': 'test_match',
            'remote_path': 'test/path.mkv'
        })
        assert r.status_code == 200
        assert r.json()['state'] == 'queued'

def test_verified_asset_skips_download(state):
    from retrieval.acquisition import queue_download
    # Pre-register a verified local asset
    with db.connect() as c:
        c.execute('''INSERT INTO local_assets (id, source_asset_id, local_path, verified)
                     VALUES (?,?,?,?)''', ('la1', 'asset_v', '/tmp/test.mkv', 1))
    r = queue_download('asset_v', 'match_v', 'path/v.mkv')
    assert r['state'] == 'already_verified'

# ---------------------------------------------------------------------------
# M1C — Generic target-player tests
# ---------------------------------------------------------------------------

def test_non_marcelo_target_no_inherited_cutoff(state):
    """A non-Marcelo target must not inherit PSG cutoff or Marcelo-specific text."""
    j = seed_job(state, target='Messi', job_id='messi_job')
    assert j['target'] == 'Messi'
    assert j['config']['reference_cutoff'] is None
    # Verify no Marcelo-specific events
    with db.connect() as c:
        events = c.execute("SELECT * FROM events WHERE job='messi_job'").fetchall()
    assert not any('Marcelo' in (e['data'] or '') for e in events)

def test_generic_target_in_api(state):
    """API must reflect generic player name, not hardcoded Marcelo."""
    seed_job(state, target='Neymar', job_id='neymar_job')
    from retrieval.api import app
    with TestClient(app) as c:
        j = c.get('/api/jobs/neymar_job').json()
        assert j['target'] == 'Neymar'

def test_job_asset_linking(state):
    from retrieval.catalog import _match_id, link_existing_jobs
    source = make_video(state)
    j = {'id': 'link_job', 'path': str(source), 'target': 'Test', 'filename': 'source.mp4',
         'match': 'Test', 'input_hash': 'abc', 'width': 320, 'height': 180,
         'fps': 25, 'duration': 12, 'start': 0, 'end': 12, 'status': 'ready',
         'stage': 'Ready', 'elapsed': 1, 'config': {}, 'versions': {}, 'created': 1}
    db.put('jobs', 'link_job', j)
    # Register local asset
    with db.connect() as c:
        c.execute('''INSERT INTO local_assets (id, local_path, verified)
                     VALUES (?,?,?)''', ('la_link', str(source), 1))
    linked = link_existing_jobs()
    assert linked >= 1

# ---------------------------------------------------------------------------
# M1D — Activity moment tests
# ---------------------------------------------------------------------------

def test_create_moment_requires_accepted_identity(state):
    """Cannot label activity without accepted identity."""
    from retrieval.activity import create_moment
    seed_job(state)
    with pytest.raises(ValueError, match='accepted'):
        create_moment('job', 'job-t0', 0.2, 1.0, 'walking')

def test_create_and_list_moments(state):
    from retrieval.activity import create_moment, list_moments
    seed_job(state)
    # Accept identity first
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    m = create_moment('job', 'job-t0', 0.2, 0.8, 'walking', target_player='Marcelo')
    assert m['activity'] == 'walking'
    assert m['status'] == 'confirmed'
    moments = list_moments('job')
    assert len(moments) == 1
    assert moments[0]['id'] == m['id']

def test_moment_overlap_rejected(state):
    from retrieval.activity import create_moment
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    create_moment('job', 'job-t0', 0.2, 0.6, 'walking')
    with pytest.raises(ValueError, match='overlap'):
        create_moment('job', 'job-t0', 0.4, 0.8, 'jogging')

def test_split_moment(state):
    from retrieval.activity import create_moment, split_moment, list_moments
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    m = create_moment('job', 'job-t0', 0.2, 0.8, 'walking')
    parts = split_moment(m['id'], 0.5)
    assert len(parts) == 2
    assert parts[0]['end_time'] == 0.5
    assert parts[1]['start_time'] == 0.5
    assert parts[0]['activity'] == 'walking'
    assert parts[1]['activity'] == 'unclear'
    # Original invalidated
    all_moments = list_moments('job', include_invalidated=True)
    invalidated = [m for m in all_moments if m['status'] == 'invalidated']
    assert len(invalidated) == 1

def test_split_boundary_not_outside(state):
    from retrieval.activity import create_moment, split_moment
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    m = create_moment('job', 'job-t0', 0.2, 0.8, 'jogging')
    with pytest.raises(ValueError, match='within'):
        split_moment(m['id'], 0.1)  # Before start
    with pytest.raises(ValueError, match='within'):
        split_moment(m['id'], 0.9)  # After end

def test_relabel_moment(state):
    from retrieval.activity import create_moment, update_moment, list_moments
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    m = create_moment('job', 'job-t0', 0.2, 0.8, 'walking')
    updated = update_moment(m['id'], activity='sprinting')
    assert updated['activity'] == 'sprinting'
    assert updated['revision'] == 2
    moments = list_moments('job')
    assert moments[0]['activity'] == 'sprinting'

def test_invalid_activity_rejected(state):
    from retrieval.activity import create_moment
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    with pytest.raises(ValueError, match='Activity must'):
        create_moment('job', 'job-t0', 0.2, 0.8, 'running')

def test_identity_rejection_invalidates_moments(state):
    from retrieval.activity import create_moment, list_moments, invalidate_moments_for_tracklet
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    create_moment('job', 'job-t0', 0.2, 0.6, 'walking')
    # Reject identity
    invalidate_moments_for_tracklet('job', 'job-t0')
    moments = list_moments('job')  # Only confirmed
    assert len(moments) == 0
    all_m = list_moments('job', include_invalidated=True)
    assert len(all_m) == 1
    assert all_m[0]['status'] == 'identity_invalidated'

def test_mixed_and_unclear_excluded_from_export(state):
    from retrieval.activity import create_moment, export_activity_collection
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    review('job', 'job-t1', 'accepted', 1.2, 2.0)
    review('job', 'job-t2', 'accepted', 2.2, 3.0)
    create_moment('job', 'job-t0', 0.2, 0.6, 'standing_other')
    create_moment('job', 'job-t1', 1.2, 1.8, 'mixed')
    create_moment('job', 'job-t2', 2.2, 2.8, 'unclear')
    with pytest.raises(ValueError, match='No confirmed'):
        export_activity_collection('job')

def test_empty_category_folders(state):
    """Export must create all three folders even when some are empty."""
    from retrieval.activity import create_moment, export_activity_collection
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    create_moment('job', 'job-t0', 0.2, 0.8, 'walking')
    rel = export_activity_collection('job')
    export_dir = state / rel
    assert (export_dir / 'walking').exists()
    assert (export_dir / 'jogging').exists()
    assert (export_dir / 'sprinting').exists()
    walking_clips = list((export_dir / 'walking').glob('*.mp4'))
    assert len(walking_clips) == 1
    assert len(list((export_dir / 'jogging').glob('*.mp4'))) == 0
    assert len(list((export_dir / 'sprinting').glob('*.mp4'))) == 0
    # Check manifests
    assert (export_dir / 'moments.csv').exists()
    assert (export_dir / 'moments.json').exists()
    manifest = json.loads((export_dir / 'moments.json').read_text())
    assert manifest['counts']['walking'] == 1
    assert manifest['counts']['jogging'] == 0
    assert manifest['counts']['sprinting'] == 0

def test_export_manifest_provenance(state):
    """Manifest must include source path, hash, fps, crop policy."""
    from retrieval.activity import create_moment, export_activity_collection
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    create_moment('job', 'job-t0', 0.2, 0.8, 'sprinting')
    rel = export_activity_collection('job')
    manifest = json.loads((state / rel / 'moments.json').read_text())
    assert 'source_hash' in manifest
    assert 'source_path' in manifest
    assert manifest['crop_policy']['padding'] == 0.125
    m = manifest['moments'][0]
    assert m['activity'] == 'sprinting'
    assert m['source_fps'] == 25.0
    assert m['start_time'] == 0.2
    assert m['end_time'] == 0.8
    # CSV too
    with open(state / rel / 'moments.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]['activity'] == 'sprinting'

def test_moment_api_routes(state):
    from retrieval.api import app
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    with TestClient(app) as c:
        # Create moment
        r = c.post('/api/jobs/job/moments', json={
            'tracklet_id': 'job-t0', 'start_time': 0.2,
            'end_time': 0.8, 'activity': 'walking'
        })
        assert r.status_code == 200
        mid = r.json()['id']
        # List moments
        r = c.get('/api/jobs/job/moments')
        assert r.status_code == 200
        assert len(r.json()) == 1
        # Update moment
        r = c.patch(f'/api/moments/{mid}', json={'activity': 'jogging'})
        assert r.status_code == 200
        assert r.json()['activity'] == 'jogging'
        # Delete moment
        r = c.delete(f'/api/moments/{mid}')
        assert r.status_code == 200
        r = c.get('/api/jobs/job/moments')
        assert len(r.json()) == 0  # Soft-deleted, not shown by default

def test_undo_moment(state):
    from retrieval.activity import create_moment, undo_last_moment, list_moments
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    m = create_moment('job', 'job-t0', 0.2, 0.6, 'walking')
    undo_last_moment('job')
    assert len(list_moments('job')) == 0
    undo_last_moment('job')  # Undo the undo
    assert len(list_moments('job')) == 1

def test_activity_stats_in_metrics(state):
    from retrieval.api import app
    seed_job(state)
    from retrieval.identity import review
    review('job', 'job-t0', 'accepted', 0.2, 1.0)
    from retrieval.activity import create_moment
    create_moment('job', 'job-t0', 0.2, 0.6, 'walking')
    with TestClient(app) as c:
        r = c.get('/api/jobs/job/metrics')
        assert r.status_code == 200
        m = r.json()
        assert m['activity_moments']['walking'] == 1
        assert m['total_moments'] == 1
