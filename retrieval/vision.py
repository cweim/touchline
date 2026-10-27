import hashlib,json,subprocess
from pathlib import Path
from typing import Protocol
import cv2,numpy as np,torch
from .storage import ROOT,MODELS,sha_file

class Detector(Protocol):
 def predict(self,images:list[np.ndarray])->list[list[dict]]:...
class FootballDetector:
 def __init__(self,weights=None,imgsz=640,confidence=.25,device='auto'):
  from ultralytics import YOLO
  self.path=Path(weights or MODELS/'football-player-detection.pt');self.model=YOLO(str(self.path))
  self.imgsz=imgsz;self.confidence=confidence
  self.device=('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if device=='auto' else device
  torch.set_num_threads(4)
 def predict(self,images):
  results=self.model.predict(images,imgsz=self.imgsz,conf=self.confidence,verbose=False,device=self.device,agnostic_nms=True)
  output=[]
  for result in results:
   # One device-to-host transfer per image. Scalar GPU reads per box are very slow.
   data=result.boxes.data.detach().cpu().numpy()
   output.append([{'bbox':row[:4].astype(float).tolist(),'confidence':float(row[4]),'class':self.model.names[int(row[5])]} for row in data if self.model.names[int(row[5])] in {'player','goalkeeper','referee'}])
  return output

def probe(path):
 d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]));s=next(x for x in d['streams'] if x['codec_type']=='video');num,den=map(float,s['avg_frame_rate'].split('/'))
 return {'duration':float(d['format']['duration']),'fps':num/den,'width':s['width'],'height':s['height'],'codec':s['codec_name']}
def frame_at(cap,t):
 cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,f=cap.read()
 if not ok:raise ValueError(f'Cannot decode source frame at {t:.2f}s')
 return f
def green_fraction(img):
 hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV);return float(cv2.inRange(hsv,np.array([25,35,30]),np.array([95,255,255])).mean()/255)
def wide_decision(dets,images,duration):
 counts=[sum(d['class']=='player' for d in ds) for ds in dets]
 heights=[(d['bbox'][3]-d['bbox'][1])/im.shape[0] for ds,im in zip(dets,images) for d in ds if d['class']=='player']
 pitch=float(np.median([green_fraction(im) for im in images]));count=float(np.median(counts));height=float(np.median(heights)) if heights else 1.
 reasons=[]
 if duration<2:reasons.append('Shot is shorter than 2 seconds')
 if count<6:reasons.append('Fewer than six players in representative frames')
 if height>=.25:reasons.append('Players are too large for a wide shot')
 if pitch<.32:reasons.append('Insufficient visible pitch')
 # A graphic/transition is often isolated into a short partition. Also reject highly
 # inconsistent representatives instead of trusting one crowded frame.
 if max(counts,default=0)>=6 and min(counts,default=0)<2:reasons.append('Inconsistent view / possible graphic or replay transition')
 score=float(np.clip(min(count/10,1)*(1-min(height,.5))*pitch,0,1))
 return not reasons,score,reasons or ['Wide shot: player count, scale, pitch and duration passed'],{'median_players':count,'median_height_fraction':height,'pitch_fraction':pitch}
def padded_crop(img,box,padding=.125):
 x1,y1,x2,y2=box;w=x2-x1;h=y2-y1
 x1=max(0,int(x1-w*padding));x2=min(img.shape[1],int(x2+w*padding));y1=max(0,int(y1-h*padding));y2=min(img.shape[0],int(y2+h*padding))
 if x2<=x1 or y2<=y1:raise ValueError('Empty player box')
 return img[y1:y2,x1:x2].copy()
def sharpness(crop):return float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
def quality(d,img,others):
 x1,y1,x2,y2=d['bbox'];h=y2-y1;w=x2-x1;area=max(w*h,1);overlap=0.
 for other in others:
  if other is d:continue
  a,b,c,e=other['bbox'];inter=max(0,min(x2,c)-max(x1,a))*max(0,min(y2,e)-max(y1,b));overlap=max(overlap,inter/area)
 crop=padded_crop(img,d['bbox']);blur=sharpness(crop)
 return float(np.clip(min(h/110,1)*min(blur/90,1)*d['confidence']*(1-overlap),0,1)),overlap,blur
