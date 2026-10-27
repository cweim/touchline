"""Fetch official checkpoints into this project. Run with --dinov2 for the baseline."""
import argparse,hashlib,json,subprocess,tarfile,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];models=ROOT/'models';models.mkdir(exist_ok=True)
def download(url,path):
 if path.exists():return
 temp=path.with_suffix(path.suffix+'.download');subprocess.run(['curl','--fail','-L',url,'-o',str(temp)],check=True);temp.replace(path)
p=argparse.ArgumentParser();p.add_argument('--dinov2',action='store_true');p.add_argument('--pose',action='store_true');a=p.parse_args()
download('https://drive.usercontent.google.com/download?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q&export=download&confirm=t',models/'football-player-detection.pt')
download('https://drive.usercontent.google.com/download?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x&export=download&confirm=t',models/'osnet_x1_0_msmt17.pth')
if a.pose:
 download('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose.pt',models/'yolov8x-pose.pt')
if a.dinov2:
 download('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',models/'dinov2_vits14_pretrain.pth')
 if not (models/'dinov2').exists():
  archive=models/'dinov2-source.tar.gz';download('https://codeload.github.com/facebookresearch/dinov2/tar.gz/refs/heads/main',archive)
  with tempfile.TemporaryDirectory(dir=models) as tmp:
   with tarfile.open(archive) as tar:tar.extractall(tmp,filter='data')
   shutil.move(str(next(Path(tmp).iterdir())),models/'dinov2')
  archive.unlink()
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in models.iterdir() if p.suffix in {'.pt','.pth'}};(models/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
