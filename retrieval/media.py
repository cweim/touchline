"""On-demand previews and exports, always in original-video coordinates."""
import csv,json,math,subprocess,zipfile
from pathlib import Path
import cv2,numpy as np
from . import storage as db

def original_clip(job,start,end,path):
 if not Path(job['path']).is_file():raise ValueError('Source video is unavailable; reconnect its drive')
 if not path.exists():
  temp=path.with_name(path.stem+'.tmp.mp4')
  subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',job['path'],'-t',str(end-start),'-map','0:v:0','-map','0:a:0?','-c:v','libx264','-preset','veryfast','-crf','22','-c:a','aac','-movflags','+faststart',str(temp)],check=True)
  temp.replace(path)
 return path

def crop_clip(job,track,start,end,path):
 if not Path(job['path']).is_file():raise ValueError('Source video is unavailable; reconnect its drive')
 if path.exists():return path
 cap=cv2.VideoCapture(job['path']);first=math.ceil(start*job['fps']);last=math.ceil(end*job['fps']);cap.set(cv2.CAP_PROP_POS_FRAMES,first)
 temp=path.with_name(path.stem+'.raw.mp4');writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),job['fps'],(256,512));rows=track['frames'];times=np.array([r['time'] for r in rows])
 try:
  for n in range(first,last):
   ok,img=cap.read()
   if not ok:break
   t=n/job['fps'];idx=min(max(int(np.searchsorted(times,t)),0),len(rows)-1);box=np.array(rows[idx]['bbox'])
   if idx>0 and 0<times[idx]-times[idx-1]<=.4:
    weight=float(np.clip((t-times[idx-1])/(times[idx]-times[idx-1]),0,1));box=np.array(rows[idx-1]['bbox'])*(1-weight)+box*weight
   x1,y1,x2,y2=box;w=x2-x1;h=y2-y1;x1=max(0,int(x1-.125*w));x2=min(img.shape[1],int(x2+.125*w));y1=max(0,int(y1-.125*h));y2=min(img.shape[0],int(y2+.125*h));crop=img[y1:y2,x1:x2]
   if crop.size==0:continue
   scale=min(256/crop.shape[1],512/crop.shape[0]);crop=cv2.resize(crop,(max(1,int(crop.shape[1]*scale)),max(1,int(crop.shape[0]*scale))));canvas=np.full((512,256,3),20,dtype=np.uint8);a=(512-crop.shape[0])//2;b=(256-crop.shape[1])//2;canvas[a:a+crop.shape[0],b:b+crop.shape[1]]=crop;writer.write(canvas)
 finally:writer.release();cap.release()
 encoded=path.with_name(path.stem+'.tmp.mp4');subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(temp),'-c:v','libx264','-preset','veryfast','-crf','20','-movflags','+faststart',str(encoded)],check=True);encoded.replace(path);temp.unlink();return path

def annotated_preview(job,track,start,end,path):
 if not Path(job['path']).is_file():raise ValueError('Source video is unavailable; reconnect its drive')
 if path.exists():return path
 cap=cv2.VideoCapture(job['path']);first=math.ceil(start*job['fps']);last=math.ceil(end*job['fps']);cap.set(cv2.CAP_PROP_POS_FRAMES,first)
 raw=path.with_name(path.stem+'.raw.mp4');v=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),job['fps'],(job['width'],job['height']));rows=track['frames'];times=np.array([r['time'] for r in rows])
 try:
  for n in range(first,last):
   ok,img=cap.read()
   if not ok:break
   t=n/job['fps'];i=int(np.argmin(abs(times-t)))
   if abs(times[i]-t)<=.16:
    x1,y1,x2,y2=map(int,rows[i]['bbox']);cv2.rectangle(img,(x1,y1),(x2,y2),(0,210,255),2);cv2.putText(img,f'Track {track["local_id"]}',(x1,max(25,y1-7)),0,.6,(0,210,255),2)
   cv2.putText(img,f'Source {t:.2f}s | Preview only - verify identity',(15,25),0,.6,(255,255,255),2);v.write(img)
 finally:cap.release();v.release()
 temp=path.with_name(path.stem+'.tmp.mp4');subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(raw),'-c:v','libx264','-preset','veryfast','-crf','23','-movflags','+faststart',str(temp)],check=True);temp.replace(path);raw.unlink();return path

def preview(jobid,trackid,start=None,end=None):
 job=db.get('jobs',jobid);t=db.get('tracklets',trackid)
 if t['job_id']!=jobid:raise ValueError('Tracklet belongs to another job')
 start=max(t['start'],float(start) if start is not None else t['start']);end=min(t['end'],float(end) if end is not None else min(start+8,t['end']))
 if end<=start or end-start>30:raise ValueError('Preview must be 0–30 seconds')
 path=db.jobdir(jobid)/'previews';path.mkdir(exist_ok=True)
 return annotated_preview(job,t,start,end,path/f'{trackid}-{start:.3f}-{end:.3f}-boxed.mp4')

def export_job(jobid,context=2):
 context=float(context)
 if not 0<=context<=10:raise ValueError('Context must be between 0 and 10 seconds')
 job=db.get('jobs',jobid);latest={}
 for l in db.labels(jobid):
  if l['kind']=='review':latest[l['tracklet']]=l['data']
 accepted=[(tid,x) for tid,x in latest.items() if x['status']=='accepted']
 if not accepted:raise ValueError('Accept at least one candidate before export')
 stamp=__import__('uuid').uuid4().hex[:10];directory=db.jobdir(jobid)/'exports'/stamp;directory.mkdir(parents=True,exist_ok=True);items=[]
 with db.connect() as c:scores={r['tracklet']:json.loads(r['data']) for r in c.execute('SELECT tracklet,data FROM scores WHERE job=?',(jobid,))}
 for tid,label in accepted:
  t=db.get('tracklets',tid);score=scores.get(tid,{});start,end=label['start'],label['end'];lo=max(job['start'],start-context);hi=min(job['end'],end+context)
  original_clip(job,lo,hi,directory/f'{tid}-context.mp4');crop_clip(job,t,start,end,directory/f'{tid}-player.mp4')
  image=directory/f'{tid}.jpg'
  valid=[c for c in t['crops'] if start<=c['time']<end]
  if not valid:raise ValueError('Accepted interval needs a representative crop inside its bounds')
  representative=max(valid,key=lambda c:c['quality']);__import__('shutil').copyfile(db.STATE/representative['path'],image)
  item={'target_player':job['target'],'start_time':start,'end_time':end,'duration':end-start,'match_score':score.get('score'),'tracklet_id':tid,'shot_id':t['shot_id'],'review_status':'accepted','representative_crop':image.name,'frame_level_scores':[r for r in score.get('frames',[]) if start<=r['time']<end],'bounding_boxes':[r for r in t['frames'] if start<=r['time']<end],'embedding_model_version':score.get('embedding_version'),'context_start':lo,'context_end':hi,'original_frame_clip':f'{tid}-context.mp4','cropped_player_clip':f'{tid}-player.mp4'};items.append(item)
 fields=['target_player','start_time','end_time','duration','match_score','tracklet_id','shot_id','review_status','representative_crop']
 with open(directory/'accepted.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(items)
 (directory/'accepted.json').write_text(json.dumps({'source_hash':job['input_hash'],'source_path':job['path'],'timestamp_basis':'seconds relative to original source video','settings':job['config'],'versions':job['versions'],'tracker':job.get('tracker_settings'),'cropped_clip_context':'Cropped clips contain only confirmed intervals. Original-frame clips contain requested context.','tracklets':items},indent=2))
 archive=directory.with_suffix('.zip')
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
  for p in directory.iterdir():z.write(p,p.name)
 db.event(jobid,'export',{'count':len(items),'path':str(archive)});return str(archive.relative_to(db.STATE))
