"""Streaming shot-scoped discovery; each completed shot is a resume checkpoint."""
import json,math,re,subprocess,time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,yaml
from . import storage as db
from .vision import FootballDetector,probe,frame_at,wide_decision,padded_crop,quality

def ingest(path,target='Player',start=0,end=None,reference_cutoff=None,device='auto'):
 if not target.strip():raise ValueError('Enter a target player name')
 path=Path(path).expanduser().resolve()
 if path.suffix.lower() not in {'.mp4','.mkv'} or not path.is_file():raise ValueError('Select an existing MKV or MP4 file')
 meta=probe(path);end=min(float(end) if end is not None else meta['duration'],meta['duration']);start=float(start)
 if not 0<=start<end:raise ValueError('Start must be before end and inside the video')
 fingerprint=db.sha_file(path)
 config={'fps':12.5,'imgsz':640,'confidence':.25,'scene_threshold':.12,'device':device,'min_track_duration':.5,'max_crops':8,'padding':.125,'reference_cutoff':reference_cutoff,'embedding':'osnet','likely_threshold':.8,'review_threshold':.55}
 key=__import__('hashlib').sha256(db.encode([fingerprint,start,end,target,config]).encode()).hexdigest()[:16]
 try:
  existing=db.get('jobs',key)
  if not existing.get('imported_collection'):return existing
  key=__import__('hashlib').sha256((key+'-full-discovery').encode()).hexdigest()[:16]
  try:return db.get('jobs',key)
  except KeyError:pass
 except KeyError:pass
 import ultralytics,torch
 j={'id':key,'path':str(path),'filename':path.name,'match':path.parent.name,'target':target.strip(),'input_hash':fingerprint,**meta,'start':start,'end':end,'config':config,'status':'queued','stage':'Ready to process','completed_shots':0,'total_shots':0,'elapsed':0,'created':time.time(),'updated':time.time(),'error':None,'device':None}
 j['versions']={'ultralytics':ultralytics.__version__,'torch':torch.__version__,'opencv':cv2.__version__,'detector_sha256':db.sha_file(db.MODELS/'football-player-detection.pt')}
 db.put('jobs',key,j)
 (db.jobdir(key)/'run_config.json').write_text(json.dumps(j,indent=2));return j

def segment(job):
 p=db.jobdir(job['id'])/'shot_boundaries.json'
 if p.exists():return json.loads(p.read_text())
 db.update_job(job['id'],stage='Finding camera shots')
 cmd=['ffmpeg','-hide_banner','-ss',str(job['start']),'-i',job['path'],'-t',str(job['end']-job['start']),'-vf',f"scale=640:-1,select='gt(scene,{job['config']['scene_threshold']})',showinfo",'-an','-f','null','-']
 result=subprocess.run(cmd,capture_output=True,text=True)
 if result.returncode:raise RuntimeError(result.stderr[-1500:])
 ts=[job['start']]
 for v in re.findall(r'pts_time:([0-9.]+)',result.stderr):
  t=job['start']+float(v)
  if t-ts[-1]>=.5 and t<job['end']-.1:ts.append(t)
 ts.append(job['end']);shots=[{'id':f'{job["id"]}-s{i:04}','job_id':job['id'],'index':i,'start':a,'end':b,'duration':b-a,'status':'pending','accepted':False} for i,(a,b) in enumerate(zip(ts,ts[1:]))]
 p.write_text(json.dumps(shots,indent=2))
 for s in shots:db.put('shots',s['id'],s,job['id'])
 return shots

def _materialize(job,shot,records,histories,cap):
 directory=db.jobdir(job['id']);cropdir=directory/'crops';cropdir.mkdir(exist_ok=True)
 tracks=[];writes=defaultdict(list)
 for tid,items in histories.items():
  ts=[x['time'] for x in items];heights=[x['bbox'][3]-x['bbox'][1] for x in items];widths=[x['bbox'][2]-x['bbox'][0] for x in items]
  duration=ts[-1]-ts[0]+job['sample_step']/job['fps']
  if duration<.5 or len(items)<5 or np.median(heights)<30:continue
  flags=[]
  if np.median(heights)<50:flags.append('small_player')
  if np.median([x['overlap'] for x in items])>.3:flags.append('occlusion')
  if np.median([x['sharpness'] for x in items])<20:flags.append('blur')
  if any(np.linalg.norm(np.subtract(a['bbox'][:2],b['bbox'][:2]))/max(a['bbox'][3]-a['bbox'][1],1)>2.5 for a,b in zip(items,items[1:])):flags.append('position_jump')
  selected=[]
  for row in sorted(items,key=lambda x:x['quality'],reverse=True):
   if row['overlap']>.5 or row['sharpness']<10:continue
   if all(abs(row['time']-x['time'])>=max(.25,duration/10) for x in selected):selected.append(row)
   if len(selected)>=job['config']['max_crops']:break
  if not selected:continue
  trackid=f'{shot["id"]}-t{tid:05}';crops=[]
  for k,row in enumerate(sorted(selected,key=lambda x:x['time'])):
   rel=f'jobs/{job["id"]}/crops/{trackid}-{k}.jpg'
   crops.append({'path':rel,'time':row['time'],'bbox':row['bbox'],'quality':row['quality'],'confidence':row['confidence'],'sharpness':row['sharpness'],'overlap':row['overlap']})
   writes[row['frame']].append((rel,row['bbox']))
  if len(crops)<2:flags.append('insufficient_crops')
  t={'id':trackid,'job_id':job['id'],'shot_id':shot['id'],'local_id':tid,'start':ts[0],'end':ts[-1]+job['sample_step']/job['fps'],'duration':duration,'number_of_detections':len(items),'median_detection_confidence':float(np.median([x['confidence'] for x in items])),'median_bbox_width':float(np.median(widths)),'median_bbox_height':float(np.median(heights)),'quality_score':float(np.median([x['quality'] for x in items])),'flags':flags,'crops':crops,'representative':max(crops,key=lambda x:x['quality'])['path'],'frames':items,'review_status':'unreviewed'}
  tracks.append(t)
 valid={t['local_id']:t['id'] for t in tracks}
 options=[r for r in records if sum(d.get('track_id') in valid and d['class']=='player' for d in r['detections'])>=6]
 reps=[]
 for fraction in [.25,.5,.75]:
  if not options:break
  goal=shot['start']+shot['duration']*fraction
  r=min(options,key=lambda r:abs(r['time']-goal));options=[x for x in options if abs(x['time']-r['time'])>.5]
  rel=f'jobs/{job["id"]}/frames/{shot["id"]}-{len(reps)}.jpg';(db.STATE/rel).parent.mkdir(parents=True,exist_ok=True)
  boxes=[d|{'tracklet_id':valid[d['track_id']]} for d in r['detections'] if d.get('track_id') in valid and d['class']=='player']
  reps.append({'time':r['time'],'path':rel,'boxes':boxes});writes[r['frame']].append((rel,None))
 # Decode the shot once for every selected crop. Random-seeking separately for
 # thousands of crops repeatedly decodes GOPs and is prohibitively slow on MKV.
 if writes:
  first,last=min(writes),max(writes);cap.set(cv2.CAP_PROP_POS_FRAMES,first)
  for n in range(first,last+1):
   if n in writes:
    ok,image=cap.read()
    if not ok:raise ValueError(f'Cannot decode selected frame {n}')
    for rel,box in writes[n]:cv2.imwrite(str(db.STATE/rel),padded_crop(image,box) if box else image,[cv2.IMWRITE_JPEG_QUALITY,90])
   elif not cap.grab():raise ValueError(f'Cannot decode source frame {n}')
 for t in tracks:db.put('tracklets',t['id'],t,job['id'],shot['id'])
 shot['representatives']=reps;shot['tracklets']=len(tracks)
 return shot

def process(jobid):
 job=db.get('jobs',jobid);started=time.time();prior=job.get('elapsed',0);cap=None
 try:
  if db.sha_file(job['path'])!=job['input_hash']:raise ValueError('The source file has changed. Import it as a new job.')
  db.update_job(jobid,status='processing',error=None,worker_pid=__import__('os').getpid())
  if job.get('elapsed',0)==0:db.event(jobid,'processing_start',{})
  shots=segment(job);detector=FootballDetector(imgsz=job['config']['imgsz'],confidence=job['config']['confidence'],device=job['config']['device'])
  from ultralytics.trackers.byte_tracker import BYTETracker
  from ultralytics.engine.results import Boxes
  from ultralytics.utils import ROOT
  cfg=yaml.safe_load((ROOT/'cfg/trackers/bytetrack.yaml').read_text());cfg['track_buffer']=45
  step=max(1,round(job['fps']/job['config']['fps']));job['sample_step']=step
  db.update_job(jobid,device=detector.device,sample_step=step,effective_fps=job['fps']/step,total_shots=len(shots),tracker_settings=cfg)
  cap=cv2.VideoCapture(job['path'])
  for i,initial in enumerate(shots):
   current=db.get('jobs',jobid)
   if current.get('pause_requested'):
    db.update_job(jobid,status='paused',worker_pid=None,stage='Paused — safe to resume',pause_requested=False,elapsed=prior+time.time()-started);return
   shot=db.get('shots',initial['id'])
   if shot.get('status')=='complete':continue
   db.update_job(jobid,stage=f'Inspecting shot {i+1} of {len(shots)}',completed_shots=i,elapsed=prior+time.time()-started)
   if 'wide_shot_score' not in shot:
    times=[shot['start']+shot['duration']*v for v in [.2,.5,.8]];images=[frame_at(cap,t) for t in times];dets=detector.predict(images)
    accepted,score,reasons,signals=wide_decision(dets,images,shot['duration']);shot.update(accepted=accepted,wide_shot_score=score,reasons=reasons,signals=signals)
    rel=f'jobs/{jobid}/shots/{shot["id"]}.jpg';(db.STATE/rel).parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(db.STATE/rel),images[1]);shot['representative_frame']=rel
    db.put('shots',shot['id'],shot,jobid)
   if not shot['accepted']:
    shot['status']='complete';db.put('shots',shot['id'],shot,jobid);continue
   db.update_job(jobid,stage=f'Detecting & tracking wide shot {i+1} of {len(shots)}')
   # Reuse cached raw detections when a shot was interrupted. Tracker replay is cheap.
   with db.connect() as c:
    cached={r['frame']:json.loads(r['data']) for r in c.execute('''SELECT d.frame,d.data FROM detections d JOIN jobs j ON j.id=d.job
WHERE json_extract(j.data,'$.input_hash')=? AND json_extract(j.data,'$.versions.detector_sha256')=?
AND json_extract(j.data,'$.config.imgsz')=? AND json_extract(j.data,'$.config.confidence')=?
AND d.time>=? AND d.time<?''',(job['input_hash'],job['versions']['detector_sha256'],job['config']['imgsz'],job['config']['confidence'],shot['start'],shot['end']))}
    c.execute('DELETE FROM tracklets WHERE job=? AND shot=?',(jobid,shot['id']))
   tracker=BYTETracker(SimpleNamespace(**cfg),frame_rate=job['fps']/step);records=[];histories=defaultdict(list)
   first=math.ceil(shot['start']*job['fps']/step)*step;last=math.ceil(shot['end']*job['fps']);cap.set(cv2.CAP_PROP_POS_FRAMES,first);index=first
   while index<last:
    batch=[]
    for _ in range(4):
     if index>=last:break
     ok,img=cap.read()
     if not ok:break
     batch.append((index,img))
     for skip in range(step-1):cap.grab()
     index+=step
    if not batch:break
    missing=[img for n,img in batch if n not in cached];pred=iter(detector.predict(missing) if missing else [])
    for n,img in batch:
     ds=cached.get(n)
     if ds is None:ds=next(pred)
     # Strip stale association before tracker reconstruction.
     ds=[{k:v for k,v in d.items() if k not in {'track_id'}} for d in ds]
     arr=np.array([d['bbox']+[d['confidence'],{'goalkeeper':1,'player':2,'referee':3}[d['class']]] for d in ds],dtype=np.float32).reshape(-1,6)
     result=tracker.update(Boxes(arr,img.shape[:2]),img)
     for row in result:
      tid=int(row[4]);idx=int(row[-1]);d=ds[idx];d['track_id']=tid;q,overlap,blur=quality(d,img,ds)
      histories[tid].append({'frame':n,'time':n/job['fps'],'bbox':d['bbox'],'confidence':d['confidence'],'class':d['class'],'quality':q,'overlap':overlap,'sharpness':blur})
     record={'frame':n,'time':n/job['fps'],'detections':ds};records.append(record)
     with db.connect() as c:c.execute('INSERT OR REPLACE INTO detections VALUES (?,?,?,?,?)',(jobid,shot['id'],n,n/job['fps'],db.encode(ds)))
    db.update_job(jobid,elapsed=prior+time.time()-started,current_time=min(index/job['fps'],shot['end']))
   shot=_materialize(job,shot,records,histories,cap);shot['status']='complete';db.put('shots',shot['id'],shot,jobid)
   db.update_job(jobid,completed_shots=i+1,elapsed=prior+time.time()-started)
  allshots=db.listing('shots',jobid);tracks=db.listing('tracklets',jobid)
  db.update_job(jobid,status='ready',worker_pid=None,stage='Ready to identify your player',completed_shots=len(shots),elapsed=prior+time.time()-started,tracklet_count=len(tracks),accepted_shots=sum(s['accepted'] for s in allshots))
  (db.jobdir(jobid)/'shot_boundaries.json').write_text(json.dumps(allshots,indent=2));(db.jobdir(jobid)/'run_config.json').write_text(json.dumps(db.get('jobs',jobid),indent=2))
  db.event(jobid,'processing_complete',{'tracklets':len(tracks)})
 except Exception as e:
  db.update_job(jobid,status='error',worker_pid=None,stage='Processing interrupted — resume available',error=f'{type(e).__name__}: {e}',elapsed=prior+time.time()-started)
  raise
 finally:
  if cap:cap.release()
