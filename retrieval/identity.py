"""Human labels, cached visual embeddings, and feedback-only identity rescoring."""
import hashlib,json,time
from pathlib import Path
from typing import Protocol
import cv2,numpy as np,torch
from sklearn.linear_model import LogisticRegression
from . import storage as db
from .vision import padded_crop,frame_at
class Embedder(Protocol):
 version:str
 def embed(self,paths:list[Path])->np.ndarray:...
class OSNetEmbedder:
 def __init__(self):
  from .vendor.osnet import osnet_x1_0
  path=db.MODELS/'osnet_x1_0_msmt17.pth'
  if not path.exists():raise ValueError('OSNet checkpoint missing; run scripts/setup_models.py')
  self.version='torchreid-osnet_x1_0-msmt17-'+db.sha_file(path)[:16]
  self.model=osnet_x1_0(num_classes=1041,pretrained=False)
  state=torch.load(path,map_location='cpu',weights_only=False);state=state.get('state_dict',state)
  own=self.model.state_dict();matched={k.removeprefix('module.'):v for k,v in state.items() if k.removeprefix('module.') in own and own[k.removeprefix('module.')].shape==v.shape}
  if sum(k.startswith('conv') for k in matched)<50:raise ValueError('OSNet checkpoint does not match the model')
  self.model.load_state_dict(matched,strict=False);self.model.eval();torch.set_num_threads(4)
 def embed(self,paths):return _forward(self.model,paths,(128,256))
class DINOv2Embedder:
 def __init__(self):
  directory=db.MODELS/'dinov2';weights=db.MODELS/'dinov2_vits14_pretrain.pth'
  if not directory.exists() or not weights.exists():raise ValueError('DINOv2 baseline is not installed; run scripts/setup_models.py --dinov2')
  self.version='dinov2-vits14-'+db.sha_file(weights)[:16]
  self.model=torch.hub.load(str(directory),'dinov2_vits14',source='local',pretrained=False);self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));self.model.eval()
 def embed(self,paths):return _forward(self.model,paths,(224,224))
def _forward(model,paths,size):
 vectors=[]
 with torch.inference_mode():
  for start in range(0,len(paths),32):
   images=[]
   for path in paths[start:start+32]:
    img=cv2.imread(str(path))
    if img is None:raise ValueError(f'Missing crop: {Path(path).name}')
    img=cv2.cvtColor(cv2.resize(img,size),cv2.COLOR_BGR2RGB)/255.;img=(img-np.array([.485,.456,.406]))/np.array([.229,.224,.225]);images.append(img.transpose(2,0,1))
   vec=model(torch.tensor(np.array(images),dtype=torch.float32)).numpy();vectors.extend(vec)
 a=np.array(vectors,dtype=np.float32);return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
_EMBEDDERS={}
def embedder(name='osnet'):
 if name not in _EMBEDDERS:_EMBEDDERS[name]=OSNetEmbedder() if name=='osnet' else DINOv2Embedder() if name=='dinov2' else (_ for _ in ()).throw(ValueError('Unknown embedding model'))
 return _EMBEDDERS[name]
def vectors(paths,model):
 result=[None]*len(paths);missing=[];keys=[]
 for i,path in enumerate(paths):
  key=hashlib.sha256((db.sha_file(path)+model.version).encode()).hexdigest();keys.append(key)
  with db.connect() as c:r=c.execute('SELECT vector FROM embeddings WHERE key=?',(key,)).fetchone()
  if r:result[i]=np.frombuffer(r[0],dtype=np.float32)
  else:missing.append(i)
 if missing:
  new=model.embed([paths[i] for i in missing])
  with db.connect() as c:
   for i,v in zip(missing,new):
    result[i]=v;c.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?,?)',(keys[i],model.version,np.asarray(v,dtype=np.float32).tobytes()))
 return np.array(result)
def reference_count(jobid):
 refs=[x for x in db.labels(jobid) if x['kind']=='reference'];return len({db.get('tracklets',x['tracklet'])['shot_id'] for x in refs})
def validate_portion(job,track,start,end):
 if not track['start']<=start<end<=track['end']+.001:raise ValueError('Choose a portion inside the tracklet')
 if job['config'].get('reference_cutoff') is not None and end>job['config']['reference_cutoff']:raise ValueError('References must precede the target player’s departure')
 if end-start>.0 and end-start<.5:raise ValueError('A clean reference must be at least half a second')
 if end-start>8:raise ValueError('Confirm a clean reference portion of at most eight seconds')
def save_reference(jobid,trackid,frame_time,start,end,diversity='unspecified'):
 job=db.get('jobs',jobid);t=db.get('tracklets',trackid)
 if t['job_id']!=jobid:raise ValueError('Tracklet belongs to another job')
 validate_portion(job,t,start,end)
 portion=[r for r in t['frames'] if start<=r['time']<end]
 if any(b['time']-a['time']>.5 for a,b in zip(portion,portion[1:])):raise ValueError('This portion crosses a detection gap. Trim to a continuously visible portion.')
 if not start<=frame_time<end:raise ValueError('Selected frame must lie within the clean portion')
 current=db.labels(jobid)
 if any(x['kind']=='reference' and db.get('tracklets',x['tracklet'])['shot_id']==t['shot_id'] for x in current):raise ValueError('This shot already has a reference. Choose a different camera shot.')
 if 'position_jump' in t['flags']: # Human can confirm a trimmed part, but not the complete suspicious track.
  if start<=t['start']+.01 and end>=t['end']-.01:raise ValueError('This track has position jumps. Trim to a stable portion.')
 with db.connect() as c:records=c.execute('SELECT time,data FROM detections WHERE job=? AND shot=? ORDER BY ABS(time-?) LIMIT 1',(jobid,t['shot_id'],frame_time)).fetchone()
 if records is None or abs(records['time']-frame_time)>.12:raise ValueError('Select an actual detected frame')
 ds=json.loads(records['data']);selected=next((d for d in ds if d.get('track_id')==t['local_id']),None)
 if selected is None:raise ValueError('Selected player is not tracked in this frame')
 cap=cv2.VideoCapture(job['path']);images=[];negative=[]
 try:
  # Always include the clicked crop, then temporally spaced clean body samples.
  samples=[min(t['frames'],key=lambda x:abs(x['time']-frame_time))]
  for row in sorted(t['frames'],key=lambda x:x['quality'],reverse=True):
   if start<=row['time']<end and row['overlap']<.5 and all(abs(row['time']-v['time'])>=.3 for v in samples):samples.append(row)
   if len(samples)>=8:break
  group=__import__('uuid').uuid4().hex
  for i,row in enumerate(samples):
   rel=f'jobs/{jobid}/labels/{group}-pos-{i}.jpg';(db.STATE/rel).parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(db.STATE/rel),padded_crop(frame_at(cap,row['time']),row['bbox']));images.append({'path':rel,'time':row['time'],'bbox':row['bbox']})
  image=frame_at(cap,records['time'])
  for i,d in enumerate(ds):
   if d is selected or d['class'] not in {'player','goalkeeper','referee'}:continue
   if d['bbox'][3]-d['bbox'][1]<35:continue
   a,b,c,e=d['bbox'];x,y,z,w=selected['bbox'];intersection=max(0,min(c,z)-max(a,x))*max(0,min(e,w)-max(b,y))
   if intersection/max((c-a)*(e-b),1)>.3:continue
   rel=f'jobs/{jobid}/labels/{group}-neg-{i}.jpg';cv2.imwrite(str(db.STATE/rel),padded_crop(image,d['bbox']));negative.append({'path':rel,'time':records['time'],'bbox':d['bbox']})
 finally:cap.release()
 data={'start':start,'end':end,'frame_time':frame_time,'crops':images,'negative_crops':negative,'diversity':diversity,'group':group,'label':1}
 with db.connect() as c:c.execute('INSERT INTO labels(job,tracklet,kind,data,created) VALUES (?,?,?,?,?)',(jobid,trackid,'reference',db.encode(data),time.time()))
 db.event(jobid,'reference_confirmed',{'tracklet':trackid,'shot':t['shot_id']});db.update_job(jobid,scores_stale=True,review_complete=False)
 return reference_count(jobid)
def undo(jobid):
 with db.connect() as c:
  r=c.execute('SELECT id FROM labels WHERE job=? AND active=1 ORDER BY id DESC LIMIT 1',(jobid,)).fetchone()
  if r:c.execute('UPDATE labels SET active=0 WHERE id=?',(r['id'],))
 db.event(jobid,'undo',{});db.update_job(jobid,scores_stale=True,review_complete=False)
def review(jobid,trackid,status,start=None,end=None):
 if status not in {'accepted','rejected'}:raise ValueError('Choose accepted or rejected')
 t=db.get('tracklets',trackid)
 if t['job_id']!=jobid:raise ValueError('Tracklet belongs to another job')
 start=t['start'] if start is None else start;end=t['end'] if end is None else end
 if not t['start']<=start<end<=t['end']+.001:raise ValueError('Review interval must be inside this tracklet')
 crops=[x for x in t['crops'] if start<=x['time']<end]
 if status=='accepted' and ((len(crops)<2 and not (db.get('jobs',jobid).get('imported_collection') and len(t['frames'])>=2)) or 'position_jump' in t['flags'] and start<=t['start']+.01 and end>=t['end']-.01):raise ValueError('Trim or reject this unstable / insufficiently supported tracklet')
 data={'label':int(status=='accepted'),'status':status,'start':start,'end':end,'crops':crops,'review_source':'human'}
 with db.connect() as c:c.execute('INSERT INTO labels(job,tracklet,kind,data,created) VALUES (?,?,?,?,?)',(jobid,trackid,'review',db.encode(data),time.time()))
 db.event(jobid,'candidate_reviewed',{'tracklet':trackid,'status':status});db.update_job(jobid,scores_stale=True,review_complete=False)
 # Feedback trains a new classifier from cached embeddings; scoring remains explicit.
 if reference_count(jobid)>=5:train(jobid,True)
def training_rows(jobid):
 paths=[];ys=[];latest={}
 labels=db.labels(jobid)
 for l in labels:
  if l['kind']=='review':latest[l['tracklet']]=l
 for l in labels:
  if l['kind']=='reference':
   for c in l['data']['crops']:
    p=db.STATE/c['path']
    if p.exists():paths.append(p);ys.append(1)
   for c in l['data']['negative_crops']:
    p=db.STATE/c['path']
    if p.exists():paths.append(p);ys.append(0)
 for l in latest.values():
  for c in l['data']['crops']:
   p=db.STATE/c['path']
   if p.exists():paths.append(p);ys.append(l['data']['label'])
 return paths,ys
_MODELS={}
def train(jobid,allow_five=False):
 count=reference_count(jobid)
 if count<8 and not (allow_five and count>=5):raise ValueError('Confirm eight references from different shots, or explicitly continue with five')
 job=db.get('jobs',jobid);model=embedder(job['config']['embedding']);paths,ys=training_rows(jobid)
 if len(set(ys))<2:raise ValueError('Both target and non-target examples are needed')
 x=vectors(paths,model);clf=LogisticRegression(class_weight='balanced',max_iter=1500,C=10,random_state=0).fit(x,ys);profile=x[np.array(ys)==1].mean(0);profile/=max(np.linalg.norm(profile),1e-12)
 state={'coef':clf.coef_.tolist(),'intercept':clf.intercept_.tolist(),'profile':profile.tolist(),'embedding_version':model.version,'examples':len(paths),'references':count,'trained':time.time()};(db.jobdir(jobid)/'identity_model.json').write_text(json.dumps(state,indent=2));_MODELS[jobid]=(clf,profile,model);db.update_job(jobid,identity_model=state);return clf,profile,model

def score_job(jobid,allow_five=False):
 job=db.get('jobs',jobid)
 if job['status']=='processing':raise ValueError('Pause or finish discovery before identity scoring')
 db.update_job(jobid,identity_status='scoring',identity_error=None,review_complete=False)
 try:
  clf,profile,model=train(jobid,allow_five);tracks=db.listing('tracklets',jobid)
  model_signature=hashlib.sha256(db.encode([clf.coef_.tolist(),clf.intercept_.tolist(),profile.tolist(),model.version]).encode()).hexdigest()
  with db.connect() as c:already={r['tracklet'] for r in c.execute('SELECT tracklet,data FROM scores WHERE job=?',(jobid,)) if json.loads(r['data']).get('model_signature')==model_signature}
  n=len(already)
  for t in tracks:
   if t['id'] in already:continue
   current=db.get('jobs',jobid)
   if current.get('pause_requested'):db.update_job(jobid,identity_status='paused',identity_error=None,identity_progress=n,identity_total=len(tracks),pause_requested=False);return {'scored':n,'paused':True}
   paths=[db.STATE/x['path'] for x in t['crops']];v=vectors(paths,model);sim=v@profile;prob=clf.predict_proba(v)[:,1];combined=.7*prob+.3*np.clip(sim,0,1)
   flags=list(t['flags']);jump=False
   if any(b['time']-a['time']>.5 for a,b in zip(t['frames'],t['frames'][1:])):flags.append('detection_gap')
   if len(v)>2:
    pair=v@v.T;jump=float(np.percentile(pair[np.triu_indices(len(v),1)],10))<.5
   if jump:flags.append('appearance_jump')
   if any(c['overlap']>.4 for c in t['crops']):flags.append('multiple_players_in_crop')
   valid=[i for i,c in enumerate(t['crops']) if c['overlap']<=.4 and c['sharpness']>=15 and t['median_bbox_height']>=35]
   score=float(np.median(combined[valid])) if len(valid)>=2 else 0.
   if len(valid)<2:flags.append('insufficient_usable_crops')
   band='likely' if score>=job['config']['likely_threshold'] else 'uncertain' if score>=job['config']['review_threshold'] else 'unlikely'
   if flags and band=='likely':band='uncertain'
   result={'model_signature':model_signature,'tracklet_id':t['id'],'score':score,'similarity':float(np.median(sim)),'band':band,'flags':sorted(set(flags)),'embedding_version':model.version,'frames':[{'time':c['time'],'bbox':c['bbox'],'score':float(s),'similarity':float(cos),'classifier_score':float(pr),'crop_quality':c['quality'],'detection_confidence':c['confidence']} for c,s,cos,pr in zip(t['crops'],combined,sim,prob)],'scored':time.time()}
   with db.connect() as c:c.execute('INSERT OR REPLACE INTO scores VALUES (?,?,?)',(jobid,t['id'],db.encode(result)))
   n+=1
   if n%25==0:db.update_job(jobid,identity_progress=n,identity_total=len(tracks))
  db.update_job(jobid,identity_status='ready',scores_stale=False,identity_progress=n,identity_total=n);db.event(jobid,'identity_scored',{'count':n,'embedding':model.version});(db.jobdir(jobid)/'run_config.json').write_text(json.dumps(db.get('jobs',jobid),indent=2));return {'scored':n,'references':reference_count(jobid)}
 except Exception as e:
  db.update_job(jobid,identity_status='error',identity_error=str(e));raise
