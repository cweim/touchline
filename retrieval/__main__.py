import argparse,json
from .pipeline import ingest,process
from . import storage as db
p=argparse.ArgumentParser(description='Local target-player retrieval');sub=p.add_subparsers(dest='command',required=True)
a=sub.add_parser('process');a.add_argument('video');a.add_argument('--target',required=True);a.add_argument('--start',type=float,default=0);a.add_argument('--end',type=float);a.add_argument('--reference-cutoff',type=float);a.add_argument('--device',default='auto')
a=sub.add_parser('resume');a.add_argument('job_id');a=sub.add_parser('status');a.add_argument('job_id',nargs='?')
a=sub.add_parser('score');a.add_argument('job_id');a.add_argument('--allow-five',action='store_true');a=sub.add_parser('export');a.add_argument('job_id');a.add_argument('--context',type=float,default=2)
a=sub.add_parser('serve');a.add_argument('--port',type=int,default=8766)
a=sub.add_parser('catalog');a.add_argument('action',choices=['sync','scan','access','matches'])
a.add_argument('--repo',default='SoccerNet/SoccerNet_raw_HQ');a.add_argument('--branch',default='videos-720p')
a.add_argument('--team');a.add_argument('--season');a.add_argument('--search')
a=sub.add_parser('download');a.add_argument('action',choices=['list','queue','cancel','retry','start-worker','disk-space'])
a.add_argument('--match-id');a.add_argument('--asset-id');a.add_argument('--remote-path');a.add_argument('--task-id')
a=sub.add_parser('windows');a.add_argument('job_id');a.add_argument('--run-missing-pose',action='store_true')
a=sub.add_parser('dataset');a.add_argument('job_id');a.add_argument('--mode',choices=['tracklets','windows','both'],default='both');a.add_argument('--include-video',action='store_true');a.add_argument('--context',type=float,default=0)
x=p.parse_args()
if x.command=='process':
 j=ingest(x.video,x.target,x.start,x.end,x.reference_cutoff,x.device);print(j['id'],flush=True);process(j['id']);print(json.dumps(db.get('jobs',j['id']),indent=2))
elif x.command=='resume':process(x.job_id)
elif x.command=='status':print(json.dumps(db.get('jobs',x.job_id) if x.job_id else db.listing('jobs'),indent=2))
elif x.command=='score':
 from .identity import score_job
 print(score_job(x.job_id,x.allow_five))
elif x.command=='export':
 from .media import export_job
 print(export_job(x.job_id,x.context))
elif x.command=='serve':
 import uvicorn
 uvicorn.run('retrieval.api:app',host='127.0.0.1',port=x.port)
elif x.command=='catalog':
 from . import catalog
 if x.action=='sync':
  result=catalog.sync_catalogue(x.repo,x.branch)
  if result.get('synced'):catalog.scan_local_files();catalog.link_existing_jobs()
  print(json.dumps(result,indent=2))
 elif x.action=='scan':
  found=catalog.scan_local_files();linked=catalog.link_existing_jobs()
  print(json.dumps({'files':len(found),'linked_jobs':linked},indent=2))
 elif x.action=='access':
  print(json.dumps(catalog.check_repo_access(x.repo),indent=2))
 elif x.action=='matches':
  matches=catalog.list_matches(team=x.team,season=x.season,search=x.search)
  print(json.dumps(matches,indent=2))
elif x.command=='download':
 from . import acquisition
 if x.action=='list':
  print(json.dumps(acquisition.list_downloads(x.match_id),indent=2))
 elif x.action=='queue':
  if not x.asset_id or not x.match_id or not x.remote_path:
   print('Provide --asset-id, --match-id and --remote-path');exit(1)
  print(json.dumps(acquisition.queue_download(x.asset_id,x.match_id,x.remote_path),indent=2))
 elif x.action=='cancel':
  if not x.task_id:print('Provide --task-id');exit(1)
  acquisition.cancel_download(x.task_id)
 elif x.action=='retry':
  if not x.task_id:print('Provide --task-id');exit(1)
  acquisition.retry_download(x.task_id)
 elif x.action=='start-worker':
  acquisition.start_download_worker();print('Download worker started')
 elif x.action=='disk-space':
  print(json.dumps(acquisition.check_disk_space(),indent=2))

if x.command=='windows':
 from .windows import build
 result=build(x.job_id,x.run_missing_pose);print(json.dumps({'windows':len(result['windows']),'skipped':len(result['skipped'])}) if result else 'Paused')
elif x.command=='dataset':
 from .dataset import export
 print(export(x.job_id,mode=x.mode,include_video=x.include_video,context=x.context))
