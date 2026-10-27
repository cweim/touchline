"""SoccerNet catalogue synchronization and local file reconciliation."""
import json, logging, os, re, sqlite3, time
from pathlib import Path
from . import storage as db

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema migration — additive, safe alongside running workers
# ---------------------------------------------------------------------------

CATALOG_SCHEMA = '''
CREATE TABLE IF NOT EXISTS dataset_sources (
    id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    branch TEXT,
    resolved_revision TEXT,
    access_state TEXT DEFAULT 'unknown',
    last_sync REAL,
    last_sync_error TEXT,
    data TEXT
);
CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    remote_path TEXT NOT NULL,
    raw_name TEXT,
    competition TEXT,
    season TEXT,
    match_date TEXT,
    home_team TEXT,
    away_team TEXT,
    split TEXT,
    data TEXT,
    UNIQUE(source_id, remote_path)
);
CREATE TABLE IF NOT EXISTS source_assets (
    id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL,
    half INTEGER,
    representation TEXT,
    remote_path TEXT NOT NULL,
    revision TEXT,
    reported_size INTEGER,
    asset_type TEXT DEFAULT 'video',
    data TEXT,
    UNIQUE(match_id, remote_path)
);
CREATE TABLE IF NOT EXISTS local_assets (
    id TEXT PRIMARY KEY,
    source_asset_id TEXT,
    match_id TEXT,
    local_path TEXT NOT NULL,
    file_size INTEGER,
    sha256 TEXT,
    probe_data TEXT,
    verified INTEGER DEFAULT 0,
    data TEXT
);
CREATE TABLE IF NOT EXISTS download_tasks (
    id TEXT PRIMARY KEY,
    source_asset_id TEXT NOT NULL,
    match_id TEXT NOT NULL,
    remote_path TEXT NOT NULL,
    revision TEXT,
    destination TEXT NOT NULL,
    state TEXT DEFAULT 'queued',
    progress_bytes INTEGER DEFAULT 0,
    total_bytes INTEGER,
    error TEXT,
    attempts INTEGER DEFAULT 0,
    created REAL,
    updated REAL,
    data TEXT
);
CREATE TABLE IF NOT EXISTS metadata_assets (
    id TEXT PRIMARY KEY,
    match_id TEXT,
    source_id TEXT,
    asset_type TEXT,
    remote_path TEXT,
    local_path TEXT,
    parsed INTEGER DEFAULT 0,
    data TEXT
);
CREATE TABLE IF NOT EXISTS job_asset_links (
    job_id TEXT NOT NULL,
    local_asset_id TEXT NOT NULL,
    match_id TEXT,
    PRIMARY KEY(job_id, local_asset_id)
);
CREATE TABLE IF NOT EXISTS activity_moments (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    tracklet_id TEXT NOT NULL,
    target_player TEXT,
    half INTEGER,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    activity TEXT NOT NULL,
    status TEXT DEFAULT 'confirmed',
    identity_label_id INTEGER,
    shot_id TEXT,
    quality_flags TEXT,
    data TEXT,
    revision INTEGER DEFAULT 1,
    created REAL,
    updated REAL
);
CREATE TABLE IF NOT EXISTS export_snapshots (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    match_id TEXT,
    target_player TEXT,
    moment_ids TEXT,
    crop_policy TEXT,
    output_path TEXT,
    data TEXT,
    created REAL
);
'''

def migrate():
    """Run additive schema migration. Safe to call multiple times."""
    with db.connect() as c:
        c.executescript(CATALOG_SCHEMA)

migrate()  # Run on import

# ---------------------------------------------------------------------------
# Match ID generation
# ---------------------------------------------------------------------------

def _match_id(source_id: str, remote_path: str) -> str:
    """Stable match ID from source + canonical path."""
    import hashlib
    return hashlib.sha256(f'{source_id}:{remote_path}'.encode()).hexdigest()[:16]

def _asset_id(match_id: str, remote_path: str) -> str:
    import hashlib
    return hashlib.sha256(f'{match_id}:{remote_path}'.encode()).hexdigest()[:16]

def _local_asset_id(path: str) -> str:
    import hashlib
    return hashlib.sha256(path.encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Parse match metadata from SoccerNet directory structure
# ---------------------------------------------------------------------------

_MATCH_RE = re.compile(
    r'(?P<date>\d{4}-\d{2}-\d{2})\s*-\s*\d{2}-\d{2}\s+'
    r'(?P<home>.+?)\s+\d+\s*-\s*\d+\s+(?P<away>.+)'
)

def _parse_match_dir(name: str) -> dict:
    """Extract metadata from a SoccerNet match directory name."""
    m = _MATCH_RE.match(name)
    if not m:
        return {'raw_name': name}
    return {
        'raw_name': name,
        'match_date': m.group('date'),
        'home_team': m.group('home').strip(),
        'away_team': m.group('away').strip(),
    }

def _parse_competition_season(path_parts: list[str]) -> dict:
    """Extract competition and season from path components."""
    result = {}
    for part in path_parts:
        if re.match(r'\d{4}-\d{4}', part):
            result['season'] = part
        elif part not in ('videos-720p', ''):
            result['competition'] = part.replace('_', ' ').replace('-', ' ').title()
    return result

# ---------------------------------------------------------------------------
# Hugging Face catalogue sync
# ---------------------------------------------------------------------------

DEFAULT_REPO = 'SoccerNet/SoccerNet_raw_HQ'
DEFAULT_BRANCH = 'videos-720p'

def check_hf_auth() -> dict:
    """Check local Hugging Face authentication status."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.whoami()
        return {'authenticated': True, 'username': info.get('name', 'unknown')}
    except Exception as e:
        return {'authenticated': False, 'error': str(e)}

def check_repo_access(repo_id: str = DEFAULT_REPO) -> dict:
    """Check access to a specific HF dataset repository."""
    auth = check_hf_auth()
    if not auth['authenticated']:
        return {'state': 'signed_out', 'message': 'Run `hf auth login` to sign in.', **auth}
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info(repo_id)
        return {
            'state': 'approved',
            'repo_id': repo_id,
            'private': getattr(info, 'private', None),
            'gated': getattr(info, 'gated', None),
            'username': auth['username'],
        }
    except Exception as e:
        err = str(e)
        if '401' in err or 'Unauthorized' in err:
            return {'state': 'signed_out', 'message': 'Authentication expired. Run `hf auth login`.'}
        if '403' in err or 'Access' in err.lower() or 'gated' in err.lower():
            return {
                'state': 'access_required',
                'repo_id': repo_id,
                'message': f'Access required. Request access at https://huggingface.co/datasets/{repo_id}',
                'username': auth['username'],
            }
        if '404' in err:
            return {'state': 'error', 'message': f'Repository {repo_id} not found.'}
        return {'state': 'error', 'message': err}

def sync_catalogue(repo_id: str = DEFAULT_REPO, branch: str = DEFAULT_BRANCH) -> dict:
    """Sync the match catalogue from a HF repository listing. No video downloads."""
    source_id = f'{repo_id}@{branch}'
    access = check_repo_access(repo_id)
    if access['state'] not in ('approved',):
        # Save failed state
        with db.connect() as c:
            c.execute('''INSERT OR REPLACE INTO dataset_sources
                         (id, repo_id, branch, access_state, last_sync_error, data)
                         VALUES (?,?,?,?,?,?)''',
                      (source_id, repo_id, branch, access['state'],
                       access.get('message', ''), db.encode(access)))
        return {'synced': False, **access}

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # List all files in the repository on this branch
        files = list(api.list_repo_tree(repo_id, path_in_repo='', repo_type='dataset',
                                         revision=branch, recursive=True))
    except Exception as e:
        with db.connect() as c:
            c.execute('''INSERT OR REPLACE INTO dataset_sources
                         (id, repo_id, branch, access_state, last_sync_error, data)
                         VALUES (?,?,?,?,?,?)''',
                      (source_id, repo_id, branch, 'error', str(e), db.encode({})))
        return {'synced': False, 'state': 'error', 'message': str(e)}

    # Extract revision if available
    resolved_revision = None
    matches_found = 0
    assets_found = 0

    with db.connect() as c:
        # Group files by match directory
        match_dirs: dict[str, list] = {}
        for f in files:
            path = getattr(f, 'path', '') if hasattr(f, 'path') else str(f)
            rfl_path = getattr(f, 'rfilename', path) if hasattr(f, 'rfilename') else path
            size = getattr(f, 'size', None) if hasattr(f, 'size') else None
            blob_id = getattr(f, 'blob_id', None) if hasattr(f, 'blob_id') else None
            lfs = getattr(f, 'lfs', None)

            if not isinstance(path, str):
                continue

            # Video files: X_720p.mkv pattern
            if re.search(r'\d+_720p\.mkv$', path):
                parts = path.split('/')
                if len(parts) >= 2:
                    match_dir = '/'.join(parts[:-1])
                    if match_dir not in match_dirs:
                        match_dirs[match_dir] = []
                    actual_size = size
                    if lfs and hasattr(lfs, 'size'):
                        actual_size = lfs.size
                    match_dirs[match_dir].append({
                        'path': path,
                        'filename': parts[-1],
                        'size': actual_size,
                        'blob_id': blob_id,
                    })

        # Upsert matches and assets
        for match_dir, video_files in match_dirs.items():
            parts = match_dir.split('/')
            match_name = parts[-1] if parts else match_dir
            mid = _match_id(source_id, match_dir)

            meta = _parse_match_dir(match_name)
            comp_season = _parse_competition_season(parts[:-1] if len(parts) > 1 else [])

            c.execute('''INSERT INTO matches (id, source_id, remote_path, raw_name,
                         competition, season, match_date, home_team, away_team, data)
                         VALUES (?,?,?,?,?,?,?,?,?,?)
                         ON CONFLICT(source_id, remote_path) DO UPDATE SET
                         raw_name=excluded.raw_name, competition=excluded.competition,
                         season=excluded.season, match_date=excluded.match_date,
                         home_team=excluded.home_team, away_team=excluded.away_team,
                         data=excluded.data''',
                      (mid, source_id, match_dir, meta.get('raw_name', match_name),
                       comp_season.get('competition'), comp_season.get('season'),
                       meta.get('match_date'), meta.get('home_team'), meta.get('away_team'),
                       db.encode({'source_files': video_files})))
            matches_found += 1

            for vf in video_files:
                half_match = re.match(r'(\d+)_720p\.mkv', vf['filename'])
                half_num = int(half_match.group(1)) if half_match else None
                aid = _asset_id(mid, vf['path'])

                c.execute('''INSERT INTO source_assets
                             (id, match_id, half, representation, remote_path,
                              revision, reported_size, asset_type, data)
                             VALUES (?,?,?,?,?,?,?,?,?)
                             ON CONFLICT(match_id, remote_path) DO UPDATE SET
                             revision=excluded.revision, reported_size=excluded.reported_size,
                             data=excluded.data''',
                          (aid, mid, half_num, '720p', vf['path'],
                           branch, vf['size'], 'video',
                           db.encode({'blob_id': vf.get('blob_id')})))
                assets_found += 1

        # Update source record
        c.execute('''INSERT OR REPLACE INTO dataset_sources
                     (id, repo_id, branch, resolved_revision, access_state,
                      last_sync, last_sync_error, data)
                     VALUES (?,?,?,?,?,?,?,?)''',
                  (source_id, repo_id, branch, resolved_revision, 'approved',
                   time.time(), None,
                   db.encode({'matches': matches_found, 'assets': assets_found})))

    return {
        'synced': True,
        'state': 'approved',
        'matches': matches_found,
        'assets': assets_found,
        'source_id': source_id,
    }

# ---------------------------------------------------------------------------
# Local file reconciliation
# ---------------------------------------------------------------------------

def scan_local_files(video_root: str | None = None) -> list[dict]:
    """Scan configured video root for existing MKV/MP4 files and reconcile."""
    if video_root is None:
        return [item for root in db.VIDEO_ROOTS for item in scan_local_files(str(root))]

    root = Path(video_root)
    if not root.exists():
        return []

    found = []
    for p in root.rglob('*'):
        if p.suffix.lower() not in {'.mkv', '.mp4'}:
            continue
        if '.cache' in str(p) or '__pycache__' in str(p):
            continue

        rel = str(p.relative_to(db.ROOT)) if str(p).startswith(str(db.ROOT)) else str(p)
        lid = _local_asset_id(str(p))

        # Probe file
        from .vision import probe
        try:
            meta = probe(p)
        except Exception:
            meta = {}

        file_size = p.stat().st_size if p.exists() else 0

        with db.connect() as c:
            # Check if already registered
            existing = c.execute('SELECT id FROM local_assets WHERE local_path=?',
                                 (str(p),)).fetchone()
            if existing:
                found.append({'id': existing[0], 'path': str(p), 'status': 'existing'})
                continue

            # Try to match against source_assets by path pattern
            # Extract match directory and half from the local file path
            source_asset_id = None
            match_id = None

            # Look for SoccerNet path patterns
            parts = rel.replace('\\', '/').split('/')
            for i, part in enumerate(parts):
                if _MATCH_RE.match(part) and i + 1 < len(parts):
                    # Found match directory
                    filename = parts[-1]
                    # Build remote_path candidates
                    remaining = '/'.join(parts[i:])
                    # Search for matching source asset
                    row = c.execute(
                        '''SELECT sa.id, sa.match_id FROM source_assets sa
                           WHERE sa.remote_path LIKE ?''',
                        (f'%{remaining}',)).fetchone()
                    if row:
                        source_asset_id = row[0]
                        match_id = row[1]
                    else:
                        # Try matching by match dir name + filename
                        row = c.execute(
                            '''SELECT sa.id, sa.match_id FROM source_assets sa
                               JOIN matches m ON m.id = sa.match_id
                               WHERE m.raw_name = ? AND sa.remote_path LIKE ?''',
                            (part, f'%{filename}')).fetchone()
                        if row:
                            source_asset_id = row[0]
                            match_id = row[1]
                    break

            c.execute('''INSERT OR REPLACE INTO local_assets
                         (id, source_asset_id, match_id, local_path, file_size,
                          probe_data, verified, data)
                         VALUES (?,?,?,?,?,?,?,?)''',
                      (lid, source_asset_id, match_id, str(p), file_size,
                       db.encode(meta), 1 if meta else 0,
                       db.encode({'relative_path': rel})))

            found.append({
                'id': lid,
                'path': str(p),
                'source_asset_id': source_asset_id,
                'match_id': match_id,
                'size': file_size,
                'probe': meta,
                'status': 'reconciled' if source_asset_id else 'local_only',
            })

    return found

def link_existing_jobs():
    """Link existing CV jobs to local assets and matches by file path."""
    jobs = db.listing('jobs')
    linked = 0
    with db.connect() as c:
        for j in jobs:
            path = j.get('path', '')
            row = c.execute('SELECT id, match_id FROM local_assets WHERE local_path=?',
                            (path,)).fetchone()
            if row:
                la_id, mid = row[0], row[1]
                c.execute('''INSERT OR IGNORE INTO job_asset_links
                             (job_id, local_asset_id, match_id)
                             VALUES (?,?,?)''', (j['id'], la_id, mid))
                linked += 1
    return linked

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def list_sources() -> list[dict]:
    with db.connect() as c:
        rows = c.execute('SELECT * FROM dataset_sources ORDER BY last_sync DESC').fetchall()
    return [dict(r) | {'data': json.loads(r['data'] or '{}')} for r in rows]

def list_matches(source_id: str | None = None, competition: str | None = None,
                 season: str | None = None, team: str | None = None,
                 search: str | None = None) -> list[dict]:
    query = 'SELECT m.*, COUNT(sa.id) as asset_count FROM matches m LEFT JOIN source_assets sa ON sa.match_id = m.id'
    conditions = []
    params: list = []

    if source_id:
        conditions.append('m.source_id = ?')
        params.append(source_id)
    if competition:
        conditions.append('m.competition LIKE ?')
        params.append(f'%{competition}%')
    if season:
        conditions.append('m.season = ?')
        params.append(season)
    if team:
        conditions.append('(m.home_team LIKE ? OR m.away_team LIKE ?)')
        params.extend([f'%{team}%', f'%{team}%'])
    if search:
        conditions.append('(m.raw_name LIKE ? OR m.home_team LIKE ? OR m.away_team LIKE ?)')
        params.extend([f'%{search}%', f'%{search}%', f'%{search}%'])

    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    query += ' GROUP BY m.id ORDER BY m.match_date DESC, m.raw_name'

    with db.connect() as c:
        rows = c.execute(query, params).fetchall()

    result = []
    for r in rows:
        m = dict(r)
        m['data'] = json.loads(m.get('data') or '{}')
        result.append(m)
    return result

def get_match(match_id: str) -> dict:
    with db.connect() as c:
        r = c.execute('SELECT * FROM matches WHERE id=?', (match_id,)).fetchone()
        if not r:
            raise KeyError(match_id)
        m = dict(r)
        m['data'] = json.loads(m.get('data') or '{}')

        # Get source assets
        assets = c.execute('SELECT * FROM source_assets WHERE match_id=?',
                           (match_id,)).fetchall()
        m['assets'] = [dict(a) | {'data': json.loads(a['data'] or '{}')} for a in assets]

        # Get local assets
        locals_ = c.execute('SELECT * FROM local_assets WHERE match_id=?',
                            (match_id,)).fetchall()
        m['local_assets'] = [dict(la) | {
            'probe_data': json.loads(la['probe_data'] or '{}'),
            'data': json.loads(la['data'] or '{}')
        } for la in locals_]

        # Get linked jobs
        job_rows = c.execute('''SELECT j.data FROM jobs j
                                JOIN job_asset_links jal ON jal.job_id = j.id
                                WHERE jal.match_id = ?''', (match_id,)).fetchall()
        m['jobs'] = [json.loads(jr['data']) for jr in job_rows]

        # Get download tasks
        dl_rows = c.execute('SELECT * FROM download_tasks WHERE match_id=?',
                            (match_id,)).fetchall()
        m['downloads'] = [dict(d) | {'data': json.loads(d['data'] or '{}')} for d in dl_rows]

    return m

def get_match_local_state(match_id: str) -> dict:
    """Get download/local state for each half of a match."""
    with db.connect() as c:
        assets = c.execute('SELECT * FROM source_assets WHERE match_id=?',
                           (match_id,)).fetchall()
        result = {'halves': {}}
        for a in assets:
            half = a['half']
            local = c.execute(
                'SELECT * FROM local_assets WHERE source_asset_id=? AND verified=1',
                (a['id'],)).fetchone()
            dl = c.execute(
                'SELECT * FROM download_tasks WHERE source_asset_id=? ORDER BY created DESC LIMIT 1',
                (a['id'],)).fetchone()
            result['halves'][half] = {
                'source_asset_id': a['id'],
                'remote_path': a['remote_path'],
                'reported_size': a['reported_size'],
                'local': dict(local) if local else None,
                'download': dict(dl) if dl else None,
                'state': 'verified' if local else ('downloading' if dl and dl['state'] in ('queued', 'downloading') else 'not_downloaded'),
            }
    return result

def list_competitions() -> list[str]:
    with db.connect() as c:
        rows = c.execute('SELECT DISTINCT competition FROM matches WHERE competition IS NOT NULL ORDER BY competition').fetchall()
    return [r[0] for r in rows]

def list_seasons() -> list[str]:
    with db.connect() as c:
        rows = c.execute('SELECT DISTINCT season FROM matches WHERE season IS NOT NULL ORDER BY season DESC').fetchall()
    return [r[0] for r in rows]
