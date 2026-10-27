"""SQLite metadata and atomic JSON records. No source videos are copied on path import."""
import hashlib,json,os,sqlite3,time,uuid
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
_settings_path=ROOT/'local_settings.json'
SETTINGS=json.loads(_settings_path.read_text()) if _settings_path.exists() else {}
MODELS=Path(os.environ.get('TOUCHLINE_MODELS',SETTINGS.get('models_dir',str(ROOT/'models')))).expanduser().resolve()
VIDEO_ROOTS=[Path(p).expanduser().resolve() for p in SETTINGS.get('video_roots',[str(ROOT/'data')])]
STATE=Path(os.environ.get('RETRIEVAL_STATE',str(ROOT/'runtime'))).resolve()
STATE.mkdir(parents=True,exist_ok=True)
DB=STATE/'retrieval.sqlite3'

def encode(x):return json.dumps(x,separators=(',',':'),allow_nan=False)
@contextmanager
def connect():
 c=sqlite3.connect(DB,timeout=60);c.row_factory=sqlite3.Row
 c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA foreign_keys=ON')
 try:yield c;c.commit()
 except: c.rollback();raise
 finally:c.close()
def init():
 with connect() as c:
  c.executescript('''
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shots(id TEXT PRIMARY KEY,job TEXT NOT NULL,data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS shots_job ON shots(job);
CREATE TABLE IF NOT EXISTS detections(job TEXT,shot TEXT,frame INTEGER,time REAL,data TEXT,PRIMARY KEY(job,frame));
CREATE TABLE IF NOT EXISTS tracklets(id TEXT PRIMARY KEY,job TEXT NOT NULL,shot TEXT NOT NULL,data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS tracks_job ON tracklets(job);
CREATE TABLE IF NOT EXISTS labels(id INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT,tracklet TEXT,kind TEXT,data TEXT,active INTEGER DEFAULT 1,created REAL);
CREATE TABLE IF NOT EXISTS embeddings(key TEXT PRIMARY KEY,model TEXT,vector BLOB);
CREATE TABLE IF NOT EXISTS scores(job TEXT,tracklet TEXT,data TEXT,PRIMARY KEY(job,tracklet));
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT,kind TEXT,data TEXT,created REAL);
CREATE TABLE IF NOT EXISTS pose_estimates(job TEXT NOT NULL,tracklet TEXT NOT NULL,data TEXT,PRIMARY KEY(job,tracklet));
''')
def put(table,id,data,job=None,shot=None):
 if table not in {'jobs','shots','tracklets'}:raise ValueError('Invalid table')
 cols=['id'];vals=[id]
 if job is not None:cols+=['job'];vals+=[job]
 if shot is not None:cols+=['shot'];vals+=[shot]
 cols+=['data'];vals+=[encode(data)]
 with connect() as c:c.execute(f'INSERT OR REPLACE INTO {table} ({",".join(cols)}) VALUES ({",".join("?" for _ in cols)})',vals)
def get(table,id):
 if table not in {'jobs','shots','tracklets'}:raise ValueError('Invalid table')
 with connect() as c:r=c.execute(f'SELECT data FROM {table} WHERE id=?',(id,)).fetchone()
 if not r:raise KeyError(id)
 return json.loads(r[0])
def listing(table,job=None):
 if table not in {'jobs','shots','tracklets'}:raise ValueError('Invalid table')
 with connect() as c:rows=c.execute(f'SELECT data FROM {table}'+(' WHERE job=?' if job else ''),(job,) if job else ()).fetchall()
 return [json.loads(r[0]) for r in rows]
def update_job(id,**changes):
 with connect() as c:
  c.execute('BEGIN IMMEDIATE');row=c.execute('SELECT data FROM jobs WHERE id=?',(id,)).fetchone()
  if row is None:raise KeyError(id)
  j=json.loads(row[0]);j.update(changes);j['updated']=time.time();c.execute('UPDATE jobs SET data=? WHERE id=?',(encode(j),id))
 return j
def event(job,kind,data):
 with connect() as c:c.execute('INSERT INTO events(job,kind,data,created) VALUES (?,?,?,?)',(job,kind,encode(data),time.time()))
def jobdir(id):
 p=STATE/'jobs'/id;p.mkdir(parents=True,exist_ok=True);return p
def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def labels(job):
 with connect() as c:rows=c.execute('SELECT * FROM labels WHERE job=? AND active=1 ORDER BY id',(job,)).fetchall()
 return [dict(r)|{'data':json.loads(r['data'])} for r in rows]
init()

RESET_FIELDS=('identity_model','identity_status','identity_error','identity_progress','identity_total','export_status','export_error','export_path','scores_stale','review_complete','review_session_started')
def reset_enrolment(id):
 """Reset human enrolment only; keep discovery and a recoverable decision snapshot."""
 with connect() as c:
  c.execute('BEGIN IMMEDIATE');r=c.execute('SELECT data FROM jobs WHERE id=?',(id,)).fetchone()
  if r is None:raise KeyError(id)
  j=json.loads(r[0])
  if j.get('identity_status')=='scoring' or j.get('export_status')=='working':raise ValueError('Wait for scoring or export to finish before starting over')
  ids=[r[0] for r in c.execute('SELECT id FROM labels WHERE job=? AND active=1',(id,))]
  scores=[dict(r) for r in c.execute('SELECT tracklet,data FROM scores WHERE job=?',(id,))]
  now=time.time();snapshot={'label_ids':ids,'scores':scores,'job_fields':{k:j[k] for k in RESET_FIELDS if k in j}}
  cur=c.execute('INSERT INTO events(job,kind,data,created) VALUES (?,?,?,?)',(id,'enrolment_reset',encode(snapshot),now))
  c.execute('UPDATE labels SET active=0 WHERE job=?',(id,));c.execute('DELETE FROM scores WHERE job=?',(id,))
  for k in RESET_FIELDS:j.pop(k,None)
  j.update(last_reset_event_id=cur.lastrowid,review_session_started=now,review_complete=False,scores_stale=False,updated=now)
  c.execute('UPDATE jobs SET data=? WHERE id=?',(encode(j),id))
 return j

def restore_enrolment(id):
 with connect() as c:
  c.execute('BEGIN IMMEDIATE');r=c.execute('SELECT data FROM jobs WHERE id=?',(id,)).fetchone()
  if r is None:raise KeyError(id)
  j=json.loads(r[0]);reset_id=j.get('last_reset_event_id')
  if not reset_id:raise ValueError('No start-over action to undo')
  if c.execute('SELECT COUNT(*) FROM labels WHERE job=? AND active=1',(id,)).fetchone()[0] or c.execute('SELECT COUNT(*) FROM scores WHERE job=?',(id,)).fetchone()[0]:raise ValueError('Undo start over before making new selections or running retrieval')
  r=c.execute("SELECT data FROM events WHERE id=? AND job=? AND kind='enrolment_reset'",(reset_id,id)).fetchone()
  if r is None:raise ValueError('Reset history is unavailable')
  snapshot=json.loads(r[0])
  for label_id in snapshot['label_ids']:c.execute('UPDATE labels SET active=1 WHERE job=? AND id=?',(id,label_id))
  for score in snapshot['scores']:c.execute('INSERT OR REPLACE INTO scores VALUES (?,?,?)',(id,score['tracklet'],score['data']))
  for k in RESET_FIELDS:j.pop(k,None)
  j.update(snapshot['job_fields']);j.pop('last_reset_event_id',None);j['updated']=time.time();c.execute('UPDATE jobs SET data=? WHERE id=?',(encode(j),id))
 return j
