import json,threading,time,traceback
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI,HTTPException,UploadFile,File,Body,Query
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from . import storage as db
from .pipeline import ingest,process
from .identity import reference_count,save_reference,undo,review,score_job
from .media import preview,export_job
from . import catalog, acquisition, activity, windows, dataset, azure_export
@asynccontextmanager
async def lifespan(app):
 # A restart never leaves process-local tasks permanently marked as running.
 for j in db.listing('jobs'):
  if j.get('windows_status')=='working':db.update_job(j['id'],windows_status='paused',windows_error='Build interrupted by app restart; rebuild to continue with cached poses.')
  if j.get('dataset_status')=='working':db.update_job(j['id'],dataset_status='error',dataset_error='Export interrupted by app restart; prepare it again.')
  if j.get('azure_status')=='working':db.update_job(j['id'],azure_status='error',azure_error='Azure save interrupted by app restart; retry to verify or complete the saved copy.')
 yield

app=FastAPI(title='Touchline · Player datasets',lifespan=lifespan);pool=ThreadPoolExecutor(max_workers=1);tasks={};lock=threading.Lock()
def task(key,fn,*args):
 with lock:
  if key in tasks and not tasks[key].done():raise HTTPException(409,'This action is already running')
  if key.startswith('windows-'):db.update_job(args[0],windows_status='working',windows_error=None,windows_progress=0)
  if key.startswith('dataset-'):db.update_job(args[0],dataset_status='working',dataset_error=None,dataset_progress=0,dataset_total=0)
  if key.startswith('azure-'):db.update_job(args[0],azure_status='working',azure_error=None,azure_progress=0,azure_total=0)
  tasks[key]=pool.submit(fn,*args)
 return {'started':True}
@app.exception_handler(KeyError)
async def missing(req,e):
 from fastapi.responses import JSONResponse
 return JSONResponse(status_code=404,content={'detail':'Record not found'})
@app.exception_handler(ValueError)
async def invalid(req,e):
 from fastapi.responses import JSONResponse
 return JSONResponse(status_code=400,content={'detail':str(e)})

# ---------------------------------------------------------------------------
# Existing job/video routes (preserved)
# ---------------------------------------------------------------------------

class Import(BaseModel):
 path:str;target:str=Field(min_length=1,max_length=80);start:float=0;end:float|None=None;reference_cutoff:float|None=None
@app.get('/api/videos')
def videos():
 roots=[*db.VIDEO_ROOTS,db.STATE/'inputs']
 paths={p.resolve() for root in roots if root.exists() for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.mkv','.mp4'} and '.cache' not in str(p) and not p.name.startswith('._')}
 return [{'path':str(p),'name':p.parent.name+' / '+p.name} for p in sorted(paths)]
@app.get('/api/jobs')
def jobs(include_archived:bool=False):return sorted([j for j in db.listing('jobs') if include_archived or not j.get('archived')],key=lambda x:x['created'],reverse=True)
@app.post('/api/jobs')
def create(x:Import):
 path=Path(x.path);path=path if path.is_absolute() else db.ROOT/path
 j=ingest(path,x.target,x.start,x.end,x.reference_cutoff)
 if j.get('archived'):j=db.update_job(j['id'],archived=False)
 db.event(j['id'],'ui_click',{'action':'import'});return j
@app.post('/api/upload')
async def upload(file:UploadFile=File(...)):
 name=Path(file.filename or '').name
 if Path(name).suffix.lower() not in {'.mkv','.mp4'}:raise ValueError('Choose MKV or MP4')
 directory=db.STATE/'inputs';directory.mkdir(exist_ok=True);path=directory/(__import__('uuid').uuid4().hex+'-'+name)
 try:
  with path.open('wb') as f:
   while b:=await file.read(4*1024*1024):f.write(b)
 finally:await file.close()
 return {'path':str(path)}
@app.get('/api/jobs/{id}')
def job(id):
 j=db.get('jobs',id)
 if j['status']=='processing' and j.get('worker_pid'):
  try:__import__('os').kill(j['worker_pid'],0)
  except ProcessLookupError:j=db.update_job(id,status='error',worker_pid=None,stage='Interrupted — resume from checkpoint',error='The discovery process stopped before completing.')
 j['reference_count']=reference_count(id);j['storage_bytes']=sum(x.stat().st_size for x in db.jobdir(id).rglob('*') if x.is_file());return j
@app.post('/api/jobs/{id}/process')
def run(id):
 j=db.get('jobs',id)
 if j['status']=='processing':raise HTTPException(409,'Discovery is already running; pause before resuming')
 db.update_job(id,pause_requested=False);return task('process-'+id,process,id)
@app.post('/api/jobs/{id}/start-over')
def start_over(id):
 with lock:
  if any(not f.done() for key,f in tasks.items() if key in {'score-'+id,'export-'+id}):raise HTTPException(409,'Wait for scoring or export to finish')
  # Also invalidate activity moments
  activity.invalidate_moments_for_tracklet(id, '__all__')
  return db.reset_enrolment(id)
@app.post('/api/jobs/{id}/undo-start-over')
def undo_start_over(id):return db.restore_enrolment(id)
@app.post('/api/jobs/{id}/archive')
def archive(id):
 j=db.get('jobs',id)
 if j['status']=='processing' or j.get('identity_status')=='scoring' or j.get('export_status')=='working':raise ValueError('Wait for processing to finish before archiving this job')
 return db.update_job(id,archived=True)
@app.post('/api/jobs/{id}/restore')
def restore(id):return db.update_job(id,archived=False)
@app.post('/api/jobs/{id}/pause')
def pause(id):return db.update_job(id,pause_requested=True)
@app.get('/api/jobs/{id}/shots')
def shots(id):return sorted(db.listing('shots',id),key=lambda x:x['start'])
@app.get('/api/jobs/{id}/frames')
def frames(id):
 j=db.get('jobs',id);refs={db.get('tracklets',x['tracklet'])['shot_id'] for x in db.labels(id) if x['kind']=='reference'}
 session_start=j.get('review_session_started',0)
 with db.connect() as c:dismissed={json.loads(r['data']).get('shot_id') for r in c.execute("SELECT data FROM events WHERE job=? AND kind IN ('not_visible','skip') AND created>?",(id,session_start))}
 dismissed.discard(None);exclude=refs|dismissed;cutoff=j['config'].get('reference_cutoff');result=[]
 for s in sorted(db.listing('shots',id),key=lambda x:x['start']):
  if s.get('status')!='complete' or not s.get('accepted') or s['id'] in exclude:continue
  for r in s.get('representatives',[]):
   if cutoff is None or r['time']<cutoff:result.append(r|{'shot_id':s['id'],'shot_index':s['index'],'wide_shot_score':s['wide_shot_score']})
 return result
@app.get('/api/jobs/{id}/labels')
def labels(id):
 return [x|{'tracklet':db.get('tracklets',x['tracklet'])} if x['tracklet'] else x for x in db.labels(id)]
@app.get('/api/jobs/{id}/candidates')
def candidates(id):
 with db.connect() as c:scores={r['tracklet']:json.loads(r['data']) for r in c.execute('SELECT tracklet,data FROM scores WHERE job=?',(id,))}
 latest={}
 for l in db.labels(id):
  if l['kind']=='review':latest[l['tracklet']]=l['data']
 result=[]
 for t in db.listing('tracklets',id):
  lite={k:v for k,v in t.items() if k!='frames'};lite['identity']=scores.get(t['id']);lite['review_status']=latest.get(t['id'],{}).get('status','unreviewed');lite['accepted_interval']=latest.get(t['id']);result.append(lite)
 return sorted(result,key=lambda x:(x['identity'].get('score') if x['identity'].get('score') is not None else -1) if x['identity'] else -1,reverse=True)
@app.get('/api/tracklets/{id}')
def tracklet(id):return db.get('tracklets',id)
class Reference(BaseModel):
 tracklet_id:str;frame_time:float;start:float;end:float;diversity:str='unspecified'
@app.post('/api/jobs/{id}/references')
def reference(id,x:Reference):
 n=save_reference(id,x.tracklet_id,x.frame_time,x.start,x.end,x.diversity);return {'reference_count':n}
@app.post('/api/jobs/{id}/undo')
def undo_last(id):undo(id);return {'ok':True}
class Correction(BaseModel):
 tracklet_id:str;status:str;start:float|None=None;end:float|None=None
@app.post('/api/jobs/{id}/review')
def correct(id,x:Correction):
 review(id,x.tracklet_id,x.status,x.start,x.end)
 # When rejecting, invalidate dependent activity moments
 if x.status == 'rejected':
  activity.invalidate_moments_for_tracklet(id, x.tracklet_id)
 return {'saved':True,'scores_stale':True}
class AutoAccept(BaseModel):threshold:float=0.85
@app.post('/api/jobs/{id}/auto-accept')
def auto_accept(id,x:AutoAccept):
 if not 0<x.threshold<=1:raise ValueError('Threshold must be between 0 and 1')
 with db.connect() as c:scores={r['tracklet']:json.loads(r['data']) for r in c.execute('SELECT tracklet,data FROM scores WHERE job=?',(id,))}
 latest={}
 for l in db.labels(id):
  if l['kind']=='review':latest[l['tracklet']]=l['data']
 eligible=[(tid,s) for tid,s in scores.items() if s['score']>=x.threshold and tid not in latest and not s.get('flags')]
 if not eligible:return {'accepted':0,'threshold':x.threshold}
 now=time.time()
 with db.connect() as c:
  for tid,s in eligible:
   data=db.encode({'label':1,'status':'accepted','start':s['frames'][0]['time'],'end':s['frames'][-1]['time'],'crops':[],'auto':True,'review_source':'automatic'})
   c.execute('INSERT INTO labels(job,tracklet,kind,data,created) VALUES (?,?,?,?,?)',(id,tid,'review',data,now))
 db.event(id,'auto_accept',{'threshold':x.threshold,'accepted':len(eligible)})
 db.update_job(id,scores_stale=True,review_complete=False)
 return {'accepted':len(eligible),'threshold':x.threshold}
class Score(BaseModel):allow_five:bool=False
@app.post('/api/jobs/{id}/score')
def score(id,x:Score):
 count=reference_count(id)
 if count<8 and not (x.allow_five and count>=5):raise ValueError('Confirm eight independent references, or choose the five-reference exception')
 return task('score-'+id,score_job,id,x.allow_five)
@app.patch('/api/jobs/{id}/thresholds')
def thresholds(id,x:dict=Body(...)):
 j=db.get('jobs',id);lo=float(x.get('review_threshold',.55));hi=float(x.get('likely_threshold',.8))
 if not 0<=lo<hi<=1:raise ValueError('Thresholds must satisfy 0 ≤ review < likely ≤ 1')
 embedding=x.get('embedding',j['config']['embedding'])
 if embedding not in {'osnet','dinov2'}:raise ValueError('Choose OSNet or DINOv2')
 cfg=j['config']|{'review_threshold':lo,'likely_threshold':hi,'embedding':embedding};updated=db.update_job(id,config=cfg,scores_stale=True);(db.jobdir(id)/'run_config.json').write_text(json.dumps(updated,indent=2));return cfg
@app.get('/api/jobs/{id}/preview/{trackid}')
def clip(id,trackid,start:float|None=None,end:float|None=None):return FileResponse(preview(id,trackid,start,end),media_type='video/mp4')
@app.post('/api/jobs/{id}/export')
def export(id,x:dict=Body(default={})):
 def work():
  try:db.update_job(id,export_status='working',export_error=None);p=export_job(id,float(x.get('context',2)));db.update_job(id,export_status='ready',export_path=p)
  except Exception as e:db.update_job(id,export_status='error',export_error=str(e))
 return task('export-'+id,work)
@app.post('/api/jobs/{id}/event')
def event(id,x:dict=Body(...)):
 if x.get('kind') not in {'ui_click','not_visible','skip','known_appearance'}:raise ValueError('Unknown event')
 db.event(id,x['kind'],x.get('data',{}));return {'saved':True}
@app.post('/api/jobs/{id}/finish-review')
def finish_review(id):
 with db.connect() as c:n=c.execute('SELECT COUNT(*) FROM scores WHERE job=?',(id,)).fetchone()[0]
 if not n:raise ValueError('Run retrieval before completing its review')
 db.event(id,'review_complete',{});return db.update_job(id,review_complete=True)
@app.get('/api/jobs/{id}/metrics')
def metrics(id):
 j=db.get('jobs',id);tracks=db.listing('tracklets',id);latest={}
 for l in db.labels(id):
  if l['kind']=='review':latest[l['tracklet']]=l['data']
 with db.connect() as c:events=[dict(r)|{'data':json.loads(r['data'])} for r in c.execute('SELECT * FROM events WHERE job=?',(id,))];scored=c.execute('SELECT COUNT(*) FROM scores WHERE job=?',(id,)).fetchone()[0]
 events=[e for e in events if e['created']>=j.get('review_session_started',0) or e['kind']=='known_appearance']
 reviewed=sorted((v['start'],v['end']) for v in latest.values());union=[]
 for lo,hi in reviewed:
  if union and lo<=union[-1][1]:union[-1][1]=max(hi,union[-1][1])
  else:union.append([lo,hi])
 known=[e['data'] for e in events if e['kind']=='known_appearance'];accepted=[x for x in latest.values() if x['status']=='accepted'];missed=[k for k in known if not any(min(k['end'],a['end'])>max(k['start'],a['start']) for a in accepted)]
 # Include activity moment counts
 moments=activity.list_moments(id)
 moment_counts={act:sum(1 for m in moments if m['activity']==act) for act in activity.ALL_ACTIVITIES}
 return {'user_clicks':sum(e['kind']=='ui_click' for e in events),'references':reference_count(id),'candidate_tracklets':len(tracks),'scored_tracklets':scored,'accepted_tracklets':len(accepted),'rejected_tracklets':sum(x['status']=='rejected' for x in latest.values()),'false_positive_tracklets':sum(x['status']=='rejected' for x in latest.values()),'missed_known_appearances':len(missed) if known and j.get('review_complete') else None,'review_complete':bool(j.get('review_complete')),'known_appearances':len(known),'discovery_seconds':j['elapsed'],'storage_bytes':sum(x.stat().st_size for x in db.jobdir(id).rglob('*') if x.is_file()),'reviewed_footage_percent':100*sum(b-a for a,b in union)/(j['end']-j['start']),'activity_moments':moment_counts,'total_moments':len(moments),'notes':'False positives count human-rejected scored candidates; no model precision is claimed. Known misses are pending until the user completes review.'}

# ---------------------------------------------------------------------------
# M1A — Catalogue and Match Library routes
# ---------------------------------------------------------------------------

@app.get('/api/catalog/sources')
def catalog_sources():return catalog.list_sources()

@app.get('/api/catalog/access')
def catalog_access(repo_id:str=catalog.DEFAULT_REPO):return catalog.check_repo_access(repo_id)

@app.get('/api/catalog/auth')
def catalog_auth():return catalog.check_hf_auth()

@app.post('/api/catalog/sync')
def catalog_sync(x:dict=Body(default={})):
 repo=x.get('repo_id',catalog.DEFAULT_REPO);branch=x.get('branch',catalog.DEFAULT_BRANCH)
 result=catalog.sync_catalogue(repo,branch)
 if result.get('synced'):
  catalog.scan_local_files()
  catalog.link_existing_jobs()
 return result

@app.post('/api/catalog/scan-local')
def catalog_scan():
 found=catalog.scan_local_files()
 linked=catalog.link_existing_jobs()
 return {'files':found,'linked_jobs':linked}

@app.get('/api/catalog/matches')
def catalog_matches(competition:str|None=None,season:str|None=None,
                    team:str|None=None,search:str|None=None):
 return catalog.list_matches(competition=competition,season=season,team=team,search=search)

@app.get('/api/catalog/matches/{match_id}')
def catalog_match(match_id:str):return catalog.get_match(match_id)

@app.get('/api/catalog/matches/{match_id}/state')
def catalog_match_state(match_id:str):return catalog.get_match_local_state(match_id)

@app.get('/api/catalog/competitions')
def catalog_competitions():return catalog.list_competitions()

@app.get('/api/catalog/seasons')
def catalog_seasons():return catalog.list_seasons()

# ---------------------------------------------------------------------------
# M1B — Download management routes
# ---------------------------------------------------------------------------

@app.get('/api/downloads')
def list_dl(match_id:str|None=None,state:str|None=None):return acquisition.list_downloads(match_id,state)

@app.post('/api/downloads')
def queue_dl(x:dict=Body(...)):
 return acquisition.queue_download(
  source_asset_id=x['source_asset_id'],match_id=x['match_id'],
  remote_path=x['remote_path'],revision=x.get('revision'),
  destination=x.get('destination'),reported_size=x.get('reported_size'))

@app.post('/api/downloads/{task_id}/cancel')
def cancel_dl(task_id:str):acquisition.cancel_download(task_id);return {'cancelled':True}

@app.post('/api/downloads/{task_id}/retry')
def retry_dl(task_id:str):acquisition.retry_download(task_id);return {'retried':True}

@app.get('/api/downloads/disk-space')
def disk_space():return acquisition.check_disk_space()

@app.post('/api/downloads/start-worker')
def start_dl_worker():acquisition.start_download_worker();return {'started':True}

# ---------------------------------------------------------------------------
# M1D — Activity moment routes
# ---------------------------------------------------------------------------

class MomentCreate(BaseModel):
 tracklet_id:str;start_time:float;end_time:float;activity:str
 identity_label_id:int|None=None;target_player:str|None=None
 half:int|None=None;shot_id:str|None=None

@app.post('/api/jobs/{id}/moments')
def create_moment(id:str,x:MomentCreate):
 return activity.create_moment(id,x.tracklet_id,x.start_time,x.end_time,
  x.activity,x.identity_label_id,x.target_player,x.half,x.shot_id)

@app.get('/api/jobs/{id}/moments')
def list_moments(id:str,tracklet_id:str|None=None,activity_filter:str|None=None):
 return activity.list_moments(id,tracklet_id,activity_filter)

class MomentUpdate(BaseModel):
 activity:str|None=None;start_time:float|None=None;end_time:float|None=None

@app.patch('/api/moments/{moment_id}')
def update_moment(moment_id:str,x:MomentUpdate):
 changes={k:v for k,v in x.model_dump().items() if v is not None}
 return activity.update_moment(moment_id,**changes)

@app.delete('/api/moments/{moment_id}')
def delete_moment(moment_id:str):return activity.delete_moment(moment_id)

class MomentSplit(BaseModel):
 split_time:float

@app.post('/api/moments/{moment_id}/split')
def split_moment(moment_id:str,x:MomentSplit):
 return activity.split_moment(moment_id,x.split_time)

@app.post('/api/jobs/{id}/moments/undo')
def undo_moment(id:str):return activity.undo_last_moment(id)

@app.post('/api/jobs/{id}/export-activity')
def export_activity(id:str,x:dict=Body(default={})):
 def work():
  try:
   db.update_job(id,export_status='working',export_error=None)
   p=activity.export_activity_collection(id,x.get('match_id'),x.get('target_player_id'))
   db.update_job(id,export_status='ready',export_path=p)
  except Exception as e:
   db.update_job(id,export_status='error',export_error=str(e))
 return task('export-activity-'+id,work)

@app.post('/api/jobs/{id}/cleanup')
def cleanup(id):
 """Delete crop images for tracklets that are NOT accepted as the target player."""
 j=db.get('jobs',id)
 if j['status']=='processing' or j.get('identity_status')=='scoring' or j.get('export_status')=='working':raise ValueError('Wait for processing to finish first')
 latest={}
 for l in db.labels(id):
  if l['kind']=='review':latest[l['tracklet']]=l['data']
 accepted_ids={tid for tid,d in latest.items() if d.get('status')=='accepted'}
 ref_tracklets={l['tracklet'] for l in db.labels(id) if l['kind']=='reference'}
 keep=accepted_ids|ref_tracklets
 tracks=db.listing('tracklets',id);removed=0;freed=0
 for t in tracks:
  if t['id'] in keep:continue
  for c in t.get('crops',[]):
   p=db.STATE/c['path']
   if p.exists():freed+=p.stat().st_size;p.unlink();removed+=1
 db.event(id,'cleanup',{'removed_crops':removed,'freed_bytes':freed,'kept_tracklets':len(keep)})
 return {'removed_crops':removed,'freed_mb':round(freed/1048576,1),'kept_tracklets':len(keep)}

# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------

@app.post('/api/jobs/{id}/pose')
def run_pose(id):
 from pose.engine import estimate_job
 db.get('jobs',id)
 task(f'pose-{id}',estimate_job,id)
 return {'started':True}

@app.post('/api/jobs/{id}/pose/pause')
def pause_pose(id):
 db.update_job(id,pose_pause_requested=True)
 return {'paused':True}

@app.get('/api/jobs/{id}/pose')
def pose_status(id):
 j=db.get('jobs',id)
 with db.connect() as c:
  done=c.execute('SELECT COUNT(*) FROM pose_estimates WHERE job=?',(id,)).fetchone()[0]
  accepted=c.execute("SELECT COUNT(*) FROM labels WHERE job=? AND active=1 AND kind='review' AND json_extract(data,'$.status')='accepted'",(id,)).fetchone()[0]
 return {'id':id,'pose_status':j.get('pose_status','idle'),'pose_error':j.get('pose_error'),'pose_done':done,'pose_total':accepted}

@app.get('/api/jobs/{id}/pose/tracklets')
def pose_tracklets(id):
 db.get('jobs',id)
 with db.connect() as c:
  accepted=[r[0] for r in c.execute("SELECT tracklet FROM labels WHERE job=? AND active=1 AND kind='review' AND json_extract(data,'$.status')='accepted'",(id,))]
  done={r[0] for r in c.execute('SELECT tracklet FROM pose_estimates WHERE job=?',(id,))}
 result=[]
 for tid in accepted:
  t=db.get('tracklets',tid)
  rep=t['crops'][0]['path'] if t.get('crops') else None
  result.append({'id':tid,'start':t['start'],'end':t['end'],'duration':round(t['end']-t['start'],2),'shot_id':t['shot_id'],'representative':rep,'frame_count':len(t.get('frames',[])),'has_pose':tid in done})
 result.sort(key=lambda x:x['start'])
 return result

@app.get('/api/jobs/{id}/pose/{tid}')
def pose_detail(id,tid):
 with db.connect() as c:
  r=c.execute('SELECT data FROM pose_estimates WHERE job=? AND tracklet=?',(id,tid)).fetchone()
 if not r:raise HTTPException(404,'Pose data not found for this tracklet')
 pose=json.loads(r[0])
 from pose.analytics import compute_analytics
 pose['analytics']=compute_analytics(pose)
 return pose

@app.get('/api/jobs/{id}/pose/{tid}/video')
def pose_video(id,tid):
 path=db.STATE/'jobs'/id/'pose'/f'{tid}.mp4'
 if not path.exists():raise HTTPException(404,'Pose video not generated yet')
 return FileResponse(str(path),media_type='video/mp4')

# ---------------------------------------------------------------------------
# Static file mounts (must be last)
# ---------------------------------------------------------------------------

# Window and dataset APIs are deliberately before the frontend catch-all.
class BuildWindows(BaseModel):
 run_missing_pose:bool=False

@app.get('/api/jobs/{id}/windows')
def list_windows(id):
 summary=windows.status(id)
 manifest=windows.read_manifest(id)
 return summary|{'windows':manifest['windows'] if manifest else []}

@app.post('/api/jobs/{id}/windows')
def build_windows(id,x:BuildWindows):
 j=db.get('jobs',id)
 if not windows.accepted(id):raise ValueError('Accept at least one tracklet first')
 if j['status']=='processing' or j.get('identity_status')=='scoring' or j.get('pose_status')=='processing' or j.get('dataset_status')=='working':raise ValueError('Wait for the current processing action to finish')
 def work(_id):
  try:windows.build(id,x.run_missing_pose)
  except Exception as e:db.update_job(id,windows_status='error',windows_error=str(e))
 return task('windows-'+id,work,id)

@app.post('/api/jobs/{id}/windows/pause')
def pause_windows(id):
 return db.update_job(id,windows_pause_requested=True)

@app.get('/api/jobs/{id}/windows/{wid}/preview')
def window_preview(id,wid):
 return FileResponse(windows.preview(id,wid),media_type='video/mp4')

class DatasetExport(BaseModel):
 mode:str='both'
 window_ids:list[str]|None=None
 tracklet_ids:list[str]|None=None
 include_video:bool=False
 context:float=Field(default=0,ge=0,le=10,allow_inf_nan=False)

@app.post('/api/jobs/{id}/dataset')
def export_dataset(id,x:DatasetExport):
 j=db.get('jobs',id)
 if j.get('windows_status')=='working' or j.get('pose_status')=='processing' or j.get('identity_status')=='scoring':raise ValueError('Wait for processing to finish before exporting')
 dataset.selection(id,x.mode,x.window_ids,x.tracklet_ids)
 def work(_id):
  try:dataset.export(_id,x.mode,x.window_ids,x.tracklet_ids,x.include_video,x.context)
  except Exception as e:db.update_job(_id,dataset_status='error',dataset_error=str(e))
 return task('dataset-'+id,work,id)

@app.get('/api/jobs/{id}/dataset')
def dataset_status(id):
 j=db.get('jobs',id)
 stale=bool(j.get('dataset_signature') and j['dataset_signature']!=windows.signature(id))
 return {'status':j.get('dataset_status','idle'),'error':j.get('dataset_error'),'stale':stale,
         'progress':j.get('dataset_progress',0),'total':j.get('dataset_total',0),
         'bytes':j.get('dataset_bytes',0),'mode':j.get('dataset_mode'),
         'download_url':f'/api/jobs/{id}/dataset/download' if j.get('dataset_status')=='ready' and not stale else None}

@app.get('/api/jobs/{id}/dataset/download')
def download_dataset(id):
 return FileResponse(dataset.ready_path(id),media_type='application/zip',filename=f'touchline-{id}.zip')

@app.get('/api/jobs/{id}/azure-exports')
def azure_exports(id):
 return azure_export.status(id)

@app.post('/api/jobs/{id}/azure-exports')
def save_azure_export(id):
 config=azure_export.settings()
 if not config['available']:raise ValueError(config['message'])
 dataset.ready_path(id)
 def work(_id):
  try:azure_export.save(_id)
  except ValueError:pass  # The module records a credential-free failure message.
 return task('azure-'+id,work,id)

@app.get('/api/jobs/{id}/azure-exports/{exportid}/download')
def download_azure_export(id,exportid):
 chunks,record=azure_export.download(id,exportid)
 return StreamingResponse(chunks,media_type='application/zip',headers={
  'Content-Disposition':f'attachment; filename="touchline-{record["id"][:12]}.zip"',
  'Content-Length':str(record['bytes']), 'Cache-Control':'no-store'})

# Serve preview media only; databases, settings and dataset ZIPs use no static mount.
@app.get('/media/{relative:path}')
def media_file(relative:str):
 path=(db.STATE/relative).resolve()
 if not path.is_relative_to(db.STATE.resolve()) or path.suffix.lower() not in {'.jpg','.jpeg','.png','.mp4'} or not path.is_file():
  raise HTTPException(404,'Media not found')
 return FileResponse(path)

frontend=db.ROOT/'frontend/dist'
if frontend.exists():
 app.mount('/',StaticFiles(directory=str(frontend),html=True),name='dashboard')
