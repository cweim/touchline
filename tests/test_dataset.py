import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from retrieval import storage as db
from retrieval import dataset, windows


@pytest.fixture
def collection(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'STATE', tmp_path)
    monkeypatch.setattr(db, 'DB', tmp_path/'test.sqlite')
    db.init()
    source = tmp_path/'source.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 25, (160,180))
    for _ in range(200):
        writer.write(np.full((180,160,3), (30,120,30), np.uint8))
    writer.release()
    job = {'id':'job','path':str(source),'filename':'source.mp4','match':'Test match',
           'target':'Player A','input_hash':db.sha_file(source),'fps':25,'width':160,'height':180,
           'start':0,'end':8,'status':'ready','created':1,'elapsed':0,'config':{},'versions':{}}
    db.put('jobs','job',job)
    times = np.arange(75)*.08
    points = np.array([[60+k%2*15,40+k*5,.95] for k in range(17)],float)
    points[5:7,:2]=[[60,55],[75,55]]
    points[11:13,:2]=[[60,95],[75,95]]
    frames = [{'time':float(t),'bbox':[40,30,100,160],'frame':round(t*25),
               'quality':1,'overlap':0,'sharpness':100,'confidence':.95} for t in times]
    pose = {'tracklet_id':'track','model':'test-pose','source_hash':job['input_hash'],
            'frames':[{'time':float(t),'keypoints':points.tolist(),'bbox':[40,30,100,160]} for t in times]}
    track={'id':'track','job_id':'job','shot_id':'shot','local_id':1,'start':0,'end':6,
           'duration':6,'frames':frames,'crops':[],'representative':None,'flags':[]}
    db.put('tracklets','track',track,'job','shot')
    db.put('shots','shot',{'id':'shot','start':0,'end':6,'accepted':True,'status':'complete'},'job')
    with db.connect() as c:
        c.execute('INSERT INTO pose_estimates VALUES (?,?,?)',('job','track',db.encode(pose)))
    decide('accepted',0,6)
    return tmp_path


def decide(status,start,end,source='human'):
    with db.connect() as c:
        c.execute('INSERT INTO labels(job,tracklet,kind,data,created) VALUES (?,?,?,?,?)',
                  ('job','track','review',db.encode({'status':status,'label':int(status=='accepted'),
                   'start':start,'end':end,'review_source':source,'crops':[]}),1))


def test_windows_respect_review_trims_and_source_sampling(collection):
    decide('accepted',1,4)
    result=windows.build('job')
    assert [w['start'] for w in result['windows']]==[1,2]
    row=result['windows'][0]
    assert row['review_source']=='human'
    with np.load(collection/row['array_path']) as data:
        assert data['keypoints'].shape==(50,17,2)
        assert data['normalised'].shape==(50,17,2)
        assert data['times'][0]==1
        assert data['times'][-1]==pytest.approx(2.96)
        assert len(data['source_times']) < 50
        assert np.allclose(np.diff(data['times']),.04)
        assert data['interpolated'].any()
        assert np.allclose(data['normalised'][:,11:13,:].mean(axis=1),0)


def test_rebuild_reuses_arrays_and_latest_review_invalidates_every_access(collection):
    result=windows.build('job')
    path=collection/result['windows'][0]['array_path']
    stamp=path.stat().st_mtime_ns
    windows.build('job')
    assert path.stat().st_mtime_ns==stamp
    decide('rejected',0,6)
    assert windows.status('job')['stale']
    with pytest.raises(ValueError,match='stale'):
        windows.window('job',result['windows'][0]['id'])
    with pytest.raises(ValueError):
        dataset.export('job',mode='windows')


def test_windows_never_cross_shot_or_pose_gap(collection):
    db.put('shots','shot',{'id':'shot','start':0,'end':4,'accepted':True},'job')
    with db.connect() as c:
        pose=json.loads(c.execute('SELECT data FROM pose_estimates').fetchone()[0])
        pose['frames']=[f for f in pose['frames'] if not 2.08<=f['time']<3.2]
        c.execute('UPDATE pose_estimates SET data=?',(db.encode(pose),))
    result=windows.build('job')
    assert result['windows']
    assert all(w['end']<=4 for w in result['windows'])
    assert all(not(w['start']<3.2 and w['end']>2.08) for w in result['windows'])
    assert any('gap' in s['reason'].lower() for s in result['skipped'])


def test_missing_pose_is_skipped_without_inference(collection,monkeypatch):
    with db.connect() as c:c.execute('DELETE FROM pose_estimates')
    import pose.engine
    monkeypatch.setattr(pose.engine,'load_model',lambda:pytest.fail('Unexpected inference'))
    result=windows.build('job')
    assert not result['windows']
    assert 'No cached pose' in result['skipped'][0]['reason']


def test_low_quality_and_mismatched_pose_are_excluded(collection):
    with db.connect() as c:
        pose=json.loads(c.execute('SELECT data FROM pose_estimates').fetchone()[0])
        pose['source_hash']='another-video'
        c.execute('UPDATE pose_estimates SET data=?',(db.encode(pose),))
    assert 'source' in windows.build('job')['skipped'][0]['reason'].lower()
    pose['source_hash']=db.get('jobs','job')['input_hash']
    for f in pose['frames']:
        for kp in f['keypoints'][11:]:kp[2]=0
    with db.connect() as c:c.execute('UPDATE pose_estimates SET data=?',(db.encode(pose),))
    result=windows.build('job')
    assert not result['windows']
    assert any('visibility' in s['reason'] for s in result['skipped'])


def test_selected_export_contains_provenance_arrays_and_real_video(collection):
    decide('accepted',0,6,'automatic')
    result=windows.build('job')
    row=result['windows'][1]
    path=dataset.export('job',mode='windows',window_ids=[row['id']],include_video=True,context=1)
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['window_count']==1
        assert manifest['source']['sha256']==db.get('jobs','job')['input_hash']
        assert 'path' not in manifest['source']
        exported=json.loads(z.read('windows.json'))
        assert exported[0]['review_source']=='automatic'
        assert exported[0]['id']==row['id']
        assert 'array_path' not in exported[0]
        assert f'windows/{row["id"]}.npz' in z.namelist()
        clip=path.parent/f'windows/{row["id"]}.mp4'
        cap=cv2.VideoCapture(str(clip))
        assert cap.get(cv2.CAP_PROP_FPS)==25
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==50
        cap.release()
        assert f'windows/{row["id"]}-context.mp4' in z.namelist()
    decide('accepted',1,4)
    with pytest.raises(ValueError,match='stale'):dataset.ready_path('job')


def test_tracklet_only_export_works_without_pose_or_source_video(collection):
    with db.connect() as c:c.execute('DELETE FROM pose_estimates')
    (collection/'source.mp4').unlink()
    path=dataset.export('job',mode='tracklets')
    assert path.exists()
    with zipfile.ZipFile(path) as z:
        assert json.loads(z.read('manifest.json'))['window_count']==0
        assert len(json.loads(z.read('tracklets.json')))==1


def test_api_validates_selection_and_blocks_private_files(collection):
    result=windows.build('job')
    from retrieval.api import app
    with TestClient(app) as client:
        assert client.get('/api/jobs/job/windows').json()['count']==5
        assert client.post('/api/jobs/job/dataset',json={'mode':'nonsense'}).status_code==400
        assert client.post('/api/jobs/job/dataset',json={'mode':'windows','window_ids':[]}).status_code==400
        assert client.post('/api/jobs/job/dataset',json={'mode':'windows','window_ids':['foreign']}).status_code==400
        assert client.get('/api/jobs/foreign/windows/'+result['windows'][0]['id']+'/preview').status_code==404
        assert client.get('/media/test.sqlite').status_code==404
        assert client.get('/api/jobs/job/dataset/download').status_code==400


def test_missing_video_never_substitutes_another_input(collection,monkeypatch):
    from pose.engine import estimate_tracklet
    import pose.engine
    monkeypatch.setattr(pose.engine,'load_model',lambda:pytest.fail('Model should not load'))
    job=db.get('jobs','job')|{'path':str(collection/'missing.mp4')}
    with pytest.raises(ValueError,match='Source video'):
        estimate_tracklet(job,db.get('tracklets','track'))


def test_cached_window_build_does_not_decode_or_load_models(collection,monkeypatch):
    import pose.engine
    monkeypatch.setattr(pose.engine,'load_model',lambda:pytest.fail('Cached build loaded model'))
    monkeypatch.setattr(cv2,'VideoCapture',lambda *args:pytest.fail('Cached build decoded video'))
    assert len(windows.build('job')['windows'])==5


def test_background_build_and_export_routes_complete(collection):
    from retrieval.api import app, tasks
    with TestClient(app) as client:
        assert client.post('/api/jobs/job/windows',json={}).status_code==200
        tasks['windows-job'].result(timeout=10)
        data=client.get('/api/jobs/job/windows').json()
        assert data['status']=='ready' and data['count']==5
        ids=[data['windows'][0]['id']]
        assert client.post('/api/jobs/job/dataset',json={'mode':'windows','window_ids':ids}).status_code==200
        tasks['dataset-job'].result(timeout=10)
        data=client.get('/api/jobs/job/dataset').json()
        assert data['status']=='ready'
        response=client.get(data['download_url'])
        assert response.status_code==200
        assert response.headers['content-type']=='application/zip'


def test_full_discovery_does_not_overwrite_imported_collection(collection,monkeypatch):
    from retrieval.pipeline import ingest
    modeldir=collection/'models'
    modeldir.mkdir()
    (modeldir/'football-player-detection.pt').write_bytes(b'test-checkpoint')
    monkeypatch.setattr(db,'MODELS',modeldir)
    original=ingest(collection/'source.mp4',target='Player A')
    db.update_job(original['id'],imported_collection=True)
    fresh=ingest(collection/'source.mp4',target='Player A')
    assert fresh['id']!=original['id']
    assert db.get('jobs',original['id'])['imported_collection']
    assert ingest(collection/'source.mp4',target='Player A')['id']==fresh['id']


def test_restart_recovers_interrupted_build_and_export(collection):
    from retrieval.api import app
    db.update_job('job',windows_status='working',dataset_status='working')
    with TestClient(app) as client:
        assert client.get('/api/jobs/job/windows').json()['status']=='paused'
        assert client.get('/api/jobs/job/dataset').json()['status']=='error'


def test_generic_upload_remains_available_after_restart_without_provider_access(collection,monkeypatch):
    from retrieval.api import app
    from retrieval import catalog
    monkeypatch.setattr(db,'VIDEO_ROOTS',[collection/'inputs'])
    monkeypatch.setattr(catalog,'check_repo_access',lambda *a,**kw:pytest.fail('Local video input contacted a dataset provider'))
    original=(collection/'source.mp4').read_bytes()
    with TestClient(app) as client:
        assert client.get('/api/videos').json()==[]
        response=client.post('/api/upload',files={'file':('my-match.mp4',original,'video/mp4')})
        assert response.status_code==200
        path=response.json()['path']
        assert Path(path).read_bytes()==original
        assert (collection/'source.mp4').read_bytes()==original
    with TestClient(app) as client:
        videos=client.get('/api/videos').json()
        assert len(videos)==1
        assert videos[0]['path']==path
        assert client.get('/api/jobs/job').status_code==200
