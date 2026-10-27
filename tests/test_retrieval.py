import csv,json
from pathlib import Path
import cv2,numpy as np,pytest
from fastapi.testclient import TestClient
from retrieval import storage as db
from retrieval.vision import wide_decision,padded_crop
from retrieval.identity import save_reference,reference_count,undo,score_job,vectors,review

@pytest.fixture
def state(tmp_path,monkeypatch):
 monkeypatch.setattr(db,'STATE',tmp_path);monkeypatch.setattr(db,'DB',tmp_path/'test.sqlite');db.init();return tmp_path

def seed(state,shots=8):
 source=state/'source.mp4';v=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),25,(320,180))
 for i in range(300):
  img=np.full((180,320,3),(30,120,30),np.uint8);
  for k in range(8):cv2.rectangle(img,(k*36+4,60),(k*36+23,99),(230,230,230) if k%2 else (30,30,150),-1)
  v.write(img)
 v.release();j={'id':'job','path':str(source),'target':'Marcelo','filename':'source.mp4','match':'Test','input_hash':db.sha_file(source),'width':320,'height':180,'fps':25,'duration':12,'start':0,'end':12,'status':'ready','stage':'Ready','elapsed':1,'config':{'reference_cutoff':10,'embedding':'osnet','likely_threshold':.8,'review_threshold':.55},'versions':{},'created':1};db.put('jobs','job',j)
 for i in range(shots):
  t=i+.2;frames=[{'frame':round((t+n*.08)*25),'time':t+n*.08,'bbox':[40,40,65,135],'confidence':.9,'quality':.9,'overlap':0,'sharpness':100} for n in range(10)];paths=[]
  for k,row in enumerate(frames[::3]):
   rel=f'crop{i}-{k}.jpg';img=np.full((100,40,3),200 if i%2 else 220,np.uint8);cv2.imwrite(str(state/rel),img);paths.append({'path':rel,'time':row['time'],'bbox':row['bbox'],'quality':.9,'confidence':.9,'overlap':0,'sharpness':100})
  tr={'id':f't{i}','job_id':'job','shot_id':f's{i}','local_id':1,'start':t,'end':t+.8,'duration':.8,'frames':frames,'crops':paths,'flags':[],'number_of_detections':10,'median_bbox_height':95,'median_bbox_width':25,'median_detection_confidence':.9,'quality_score':.9,'representative':paths[0]['path']};db.put('tracklets',tr['id'],tr,'job',tr['shot_id'])
  ds=[{'bbox':[40,40,65,135],'class':'player','confidence':.9,'track_id':1},{'bbox':[150,40,175,135],'class':'player','confidence':.9,'track_id':2}]
  with db.connect() as c:c.execute('INSERT INTO detections VALUES (?,?,?,?,?)',('job',f's{i}',round(t*25),t,db.encode(ds)))
 return j

def test_wide_filter_excludes_crowded_closeup_and_short_graphic():
 pitch=np.full((720,1280,3),(20,120,20),np.uint8)
 ds=[{'class':'player','bbox':[i*90,220,i*90+30,300]} for i in range(8)]
 assert wide_decision([ds]*3,[pitch]*3,4)[0]
 large=[d|{'bbox':[0,0,200,600]} for d in ds]
 assert not wide_decision([large]*3,[pitch]*3,4)[0]
 assert not wide_decision([ds]*3,[np.zeros_like(pitch)]*3,4)[0]
 assert not wide_decision([ds]*3,[pitch]*3,1)[0]
 assert padded_crop(pitch,[-10,-10,20,20]).shape[:2]==(23,23)

def test_reference_gates_independence_cutoff_and_undo(state):
 seed(state)
 with pytest.raises(ValueError,match='eight'):score_job('job')
 assert save_reference('job','t0',.2,.2,.92)==1
 assert reference_count('job')==1
 with pytest.raises(ValueError,match='already'):save_reference('job','t0',.2,.2,.92)
 t=db.get('tracklets','t1');t['start']=10;t['end']=11;db.put('tracklets','t1',t,'job','s1')
 with pytest.raises(ValueError,match='departure'):save_reference('job','t1',10.2,10,10.7)
 undo('job');assert reference_count('job')==0

def test_cached_identity_scoring_does_not_touch_detector(state,monkeypatch):
 seed(state)
 for i in range(8):save_reference('job',f't{i}',i+.2,i+.2,i+.92)
 import retrieval.identity as identity
 class Fake:
  version='test-visual-model';calls=0
  def embed(self,paths):
   self.calls+=len(paths);return np.array([[1.,0.,0.] if '-neg-' not in str(p) else [0.,1.,0.] for p in paths],np.float32)
 m=Fake();monkeypatch.setattr(identity,'embedder',lambda _:m)
 import retrieval.pipeline as pipeline
 monkeypatch.setattr(pipeline.FootballDetector,'predict',lambda *a:pytest.fail('Rescoring reran detection'))
 assert score_job('job')['scored']==8
 calls=m.calls;score_job('job');assert m.calls==calls
 review('job','t0','accepted');assert db.get('jobs','job')['scores_stale']
 undo('job');assert not any(l['kind']=='review' for l in db.labels('job'))

def test_export_media_and_source_times(state):
 seed(state);review('job','t0','accepted')
 from retrieval.media import export_job,preview
 rel=export_job('job',2);archive=state/rel;assert archive.exists()
 directory=archive.with_suffix('');data=json.loads((directory/'accepted.json').read_text());row=data['tracklets'][0];assert row['start_time']==.2 and row['context_start']==0
 assert row['end_time']==1 and row['context_end']==3
 assert (directory/row['cropped_player_clip']).exists()
 cap=cv2.VideoCapture(str(directory/row['cropped_player_clip']));assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==20
 assert preview('job','t0',.2,1).exists()
 with pytest.raises(ValueError):export_job('job',11)

def test_api_job_isolation_and_validation(state):
 seed(state)
 from retrieval.api import app
 with TestClient(app) as c:
  assert c.get('/api/jobs/job').status_code==200
  assert c.post('/api/jobs/job/score',json={}).status_code==400
  assert c.patch('/api/jobs/job/thresholds',json={'review_threshold':.9,'likely_threshold':.5}).status_code==400
  assert c.get('/api/tracklets/missing').status_code==404
  assert c.get('/media/../../etc/passwd').status_code!=200

def test_discovery_resume_and_shot_scoped_ids(state,monkeypatch):
 seed(state,shots=0)
 from retrieval import pipeline
 j=db.get('jobs','job');j['config'].update({'fps':12.5,'imgsz':640,'confidence':.25,'device':'cpu','max_crops':5});j['versions']={'detector_sha256':'test'};j['end']=8;db.put('jobs','job',j)
 shots=[{'id':f'job-s{i}','job_id':'job','index':i,'start':i*4,'end':i*4+4,'duration':4,'status':'pending','accepted':False} for i in range(2)]
 for s in shots:db.put('shots',s['id'],s,'job')
 monkeypatch.setattr(pipeline,'segment',lambda _:shots)
 class FakeDetector:
  calls=0;device='cpu'
  def __init__(self,**kwargs):pass
  def predict(self,images):
   self.__class__.calls+=1
   if self.__class__.calls==2:db.update_job('job',pause_requested=True)
   return [[{'bbox':[i*36+4,60,i*36+23,99],'class':'player','confidence':.95} for i in range(8)] for _ in images]
 monkeypatch.setattr(pipeline,'FootballDetector',FakeDetector)
 pipeline.process('job');assert db.get('jobs','job')['status']=='paused'
 first=[t['id'] for t in db.listing('tracklets','job')];assert len(first)==8
 db.update_job('job',pause_requested=False);pipeline.process('job');assert db.get('jobs','job')['status']=='ready'
 alltracks=db.listing('tracklets','job');assert len(alltracks)==16
 assert all(tid in {t['id'] for t in alltracks} for tid in first)
 # A repeated local tracker number is distinct after a shot boundary.
 assert len({t['id'] for t in alltracks if t['local_id']==1})==2
 calls=FakeDetector.calls;pipeline.process('job');assert FakeDetector.calls==calls

def test_start_over_preserves_discovery_and_can_be_undone(state):
 seed(state)
 save_reference('job','t0',.2,.2,.92)
 other=db.get('jobs','job')|{'id':'other','status':'processing'};db.put('jobs','other',other)
 db.put('shots','shot',{'id':'shot','start':0,'end':2},'job')
 with db.connect() as c:
  c.execute('INSERT INTO scores VALUES (?,?,?)',('job','t0',db.encode({'score':.9})))
  c.execute('INSERT INTO scores VALUES (?,?,?)',('other','other-track',db.encode({'score':.8})))
 db.update_job('job',identity_status='ready',identity_model={'version':'old'},export_path='old.zip')
 before=db.get('jobs','other');result=db.reset_enrolment('job')
 assert reference_count('job')==0 and len(db.listing('tracklets','job'))==8
 assert len(db.listing('shots','job'))==1 and db.get('jobs','other')==before
 assert 'identity_model' not in result and 'export_path' not in result
 with db.connect() as c:
  assert c.execute('SELECT COUNT(*) FROM scores WHERE job=?',('job',)).fetchone()[0]==0
  assert c.execute('SELECT COUNT(*) FROM scores WHERE job=?',('other',)).fetchone()[0]==1
 db.restore_enrolment('job');assert reference_count('job')==1
 assert db.get('jobs','job')['identity_model']['version']=='old'
 with db.connect() as c:assert c.execute('SELECT COUNT(*) FROM scores WHERE job=?',('job',)).fetchone()[0]==1
 db.update_job('job',identity_status='scoring')
 with pytest.raises(ValueError,match='finish'):db.reset_enrolment('job')

def test_archive_hides_job_and_restore_recovers_it(state):
 seed(state)
 from retrieval.api import app
 with TestClient(app) as c:
  assert c.post('/api/jobs/job/archive').status_code==200
  assert c.get('/api/jobs').json()==[]
  assert c.get('/api/jobs?include_archived=true').json()[0]['archived']
  assert c.post('/api/jobs/job/restore').status_code==200
  assert len(c.get('/api/jobs').json())==1
  db.update_job('job',status='processing')
  assert c.post('/api/jobs/job/archive').status_code==400
  assert len(c.get('/api/jobs').json())==1
