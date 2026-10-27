"""Durable download queue and Hugging Face asset acquisition."""
import json, logging, os, threading, time, uuid
from pathlib import Path
from . import storage as db
from .catalog import DEFAULT_REPO, DEFAULT_BRANCH

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download task management
# ---------------------------------------------------------------------------

def queue_download(source_asset_id: str, match_id: str, remote_path: str,
                   revision: str | None = None, destination: str | None = None,
                   reported_size: int | None = None) -> dict:
    """Queue a single file for download. Deduplicates by asset + revision."""
    if destination is None:
        destination = str(db.ROOT / 'data' / 'SoccerNet' / 'videos-720p' / remote_path)

    task_id = uuid.uuid4().hex[:16]
    now = time.time()

    with db.connect() as c:
        # Check for existing verified local asset
        existing = c.execute(
            'SELECT id FROM local_assets WHERE source_asset_id=? AND verified=1',
            (source_asset_id,)).fetchone()
        if existing:
            return {'id': existing[0], 'state': 'already_verified', 'path': destination}

        # Check for existing active download
        active = c.execute(
            '''SELECT id, state FROM download_tasks
               WHERE source_asset_id=? AND state IN ('queued','downloading')''',
            (source_asset_id,)).fetchone()
        if active:
            return {'id': active[0], 'state': active[1], 'message': 'Download already queued'}

        # Check for completed download that just needs verification
        completed = c.execute(
            '''SELECT id FROM download_tasks
               WHERE source_asset_id=? AND state='complete' ''',
            (source_asset_id,)).fetchone()
        if completed:
            return {'id': completed[0], 'state': 'complete', 'message': 'Already downloaded'}

        c.execute('''INSERT INTO download_tasks
                     (id, source_asset_id, match_id, remote_path, revision,
                      destination, state, total_bytes, created, updated, data)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                  (task_id, source_asset_id, match_id, remote_path,
                   revision or DEFAULT_BRANCH, destination, 'queued',
                   reported_size, now, now, db.encode({})))

    return {'id': task_id, 'state': 'queued', 'destination': destination}

def list_downloads(match_id: str | None = None, state: str | None = None) -> list[dict]:
    query = 'SELECT * FROM download_tasks'
    conditions = []
    params: list = []
    if match_id:
        conditions.append('match_id = ?')
        params.append(match_id)
    if state:
        conditions.append('state = ?')
        params.append(state)
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    query += ' ORDER BY created DESC'

    with db.connect() as c:
        rows = c.execute(query, params).fetchall()
    return [dict(r) | {'data': json.loads(r['data'] or '{}')} for r in rows]

def cancel_download(task_id: str):
    with db.connect() as c:
        c.execute("UPDATE download_tasks SET state='cancelled', updated=? WHERE id=? AND state IN ('queued','downloading')",
                  (time.time(), task_id))

def retry_download(task_id: str):
    with db.connect() as c:
        c.execute("UPDATE download_tasks SET state='queued', error=NULL, updated=? WHERE id=? AND state IN ('failed','cancelled')",
                  (time.time(), task_id))

# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------

_worker_lock = threading.Lock()
_worker_running = False
_worker_thread: threading.Thread | None = None

def _execute_download(task_id: str):
    """Execute a single download using HF Hub SDK."""
    with db.connect() as c:
        row = c.execute('SELECT * FROM download_tasks WHERE id=?', (task_id,)).fetchone()
        if not row:
            return
        task = dict(row)

    remote_path = task['remote_path']
    destination = Path(task['destination'])
    revision = task['revision'] or DEFAULT_BRANCH

    try:
        with db.connect() as c:
            c.execute("UPDATE download_tasks SET state='downloading', updated=? WHERE id=?",
                      (time.time(), task_id))

        from huggingface_hub import hf_hub_download

        # Download using official SDK with its own cache/recovery
        downloaded_path = hf_hub_download(
            repo_id=DEFAULT_REPO,
            filename=remote_path,
            repo_type='dataset',
            revision=revision,
            local_dir=str(db.ROOT / 'data' / 'SoccerNet' / 'videos-720p'),
            local_dir_use_symlinks=False,
        )

        if not Path(downloaded_path).exists():
            raise FileNotFoundError(f'Downloaded file not found at {downloaded_path}')

        # Verify the download
        from .vision import probe
        meta = probe(Path(downloaded_path))
        file_size = Path(downloaded_path).stat().st_size

        # Register as local asset
        from .catalog import _local_asset_id
        lid = _local_asset_id(downloaded_path)

        with db.connect() as c:
            c.execute('''INSERT OR REPLACE INTO local_assets
                         (id, source_asset_id, match_id, local_path, file_size,
                          probe_data, verified, data)
                         VALUES (?,?,?,?,?,?,?,?)''',
                      (lid, task['source_asset_id'], task['match_id'],
                       downloaded_path, file_size, db.encode(meta), 1,
                       db.encode({'download_task_id': task_id})))

            c.execute('''UPDATE download_tasks
                         SET state='complete', progress_bytes=?, updated=?
                         WHERE id=?''',
                      (file_size, time.time(), task_id))

        log.info(f'Download complete: {remote_path} -> {downloaded_path}')

    except Exception as e:
        log.error(f'Download failed for {remote_path}: {e}')
        with db.connect() as c:
            c.execute('''UPDATE download_tasks
                         SET state='failed', error=?, attempts=attempts+1, updated=?
                         WHERE id=?''',
                      (str(e), time.time(), task_id))

def _download_worker_loop():
    """Background worker that processes queued downloads one at a time."""
    global _worker_running
    while _worker_running:
        with db.connect() as c:
            row = c.execute(
                "SELECT id FROM download_tasks WHERE state='queued' ORDER BY created LIMIT 1"
            ).fetchone()

        if row:
            _execute_download(row[0])
        else:
            time.sleep(2)

def start_download_worker():
    """Start the background download worker if not already running."""
    global _worker_running, _worker_thread
    with _worker_lock:
        if _worker_running:
            return
        _worker_running = True
        _worker_thread = threading.Thread(target=_download_worker_loop, daemon=True)
        _worker_thread.start()

def stop_download_worker():
    global _worker_running
    _worker_running = False

# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------

def check_disk_space(path: str | None = None) -> dict:
    """Check available disk space at target location."""
    import shutil
    target = Path(path) if path else db.ROOT / 'data'
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(target))
    return {
        'total_bytes': usage.total,
        'used_bytes': usage.used,
        'free_bytes': usage.free,
        'free_gb': round(usage.free / (1024**3), 2),
    }

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_local_asset(local_asset_id: str) -> dict:
    """Verify a local asset with probe and optional hash."""
    with db.connect() as c:
        row = c.execute('SELECT * FROM local_assets WHERE id=?', (local_asset_id,)).fetchone()
        if not row:
            raise KeyError(local_asset_id)
        la = dict(row)

    path = Path(la['local_path'])
    if not path.exists():
        with db.connect() as c:
            c.execute('UPDATE local_assets SET verified=0 WHERE id=?', (local_asset_id,))
        return {'verified': False, 'error': 'File not found'}

    from .vision import probe
    try:
        meta = probe(path)
    except Exception as e:
        return {'verified': False, 'error': f'Probe failed: {e}'}

    file_size = path.stat().st_size

    with db.connect() as c:
        c.execute('''UPDATE local_assets SET file_size=?, probe_data=?, verified=1
                     WHERE id=?''',
                  (file_size, db.encode(meta), local_asset_id))

    return {'verified': True, 'probe': meta, 'size': file_size}
