import {useEffect, useState} from 'react';
import {ArrowRight, Check, CloudUpload, Download, Layers3, Play, RefreshCw, X} from 'lucide-react';

type Json = Record<string, any>;
const api = async (path:string, body?:Json) => {
 const r=await fetch('/api'+path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
 const data=await r.json(); if(!r.ok)throw Error(data.detail||'Request failed'); return data;
};
const fmt=(n:number)=>`${Math.floor(n/60).toString().padStart(2,'0')}:${(n%60).toFixed(2).padStart(5,'0')}`;

export function DatasetWorkspace({job,view,onNavigate}:{job:Json,view:string,onNavigate:(view:string)=>void}){
 const [data,setData]=useState<Json|null>(null),[exportData,setExportData]=useState<Json|null>(null);
 const [cloud,setCloud]=useState<Json|null>(null);
 const [selected,setSelected]=useState<Set<string>>(new Set()),[revision,setRevision]=useState('');
 const [error,setError]=useState(''),[busy,setBusy]=useState(false),[missing,setMissing]=useState(false);
 const [mode,setMode]=useState('both'),[video,setVideo]=useState(false),[context,setContext]=useState(0);
 const [preview,setPreview]=useState<Json|null>(null),[limit,setLimit]=useState(48),[showSkipped,setShowSkipped]=useState(false);
 const refresh=async()=>{
  const [w,e,a]=await Promise.all([api(`/jobs/${job.id}/windows`),api(`/jobs/${job.id}/dataset`),api(`/jobs/${job.id}/azure-exports`)]);
  setData(w);setExportData(e);setCloud(a);
 };
 useEffect(()=>{
  setData(null);setExportData(null);setCloud(null);setSelected(new Set());setRevision('');setLimit(48);setPreview(null);setError('');
  let active=true;
  const poll=async()=>{try{const [w,e,a]=await Promise.all([api(`/jobs/${job.id}/windows`),api(`/jobs/${job.id}/dataset`),api(`/jobs/${job.id}/azure-exports`)]);if(active){setData(w);setExportData(e);setCloud(a)}}catch(e:any){if(active)setError(e.message)}};
  poll();const timer=setInterval(poll,3000);return()=>{active=false;clearInterval(timer)};
 },[job.id]);
 useEffect(()=>{
  if(!data?.signature||data.signature===revision)return;
  const key=`touchline-selection-${job.id}`;
  let ids=data.windows.map((w:Json)=>w.id);
  try{const saved=JSON.parse(localStorage.getItem(key)||'null');if(saved?.signature===data.signature)ids=saved.ids}catch{}
  setSelected(new Set(ids));setRevision(data.signature);setPreview(null);
 },[data?.signature,job.id,revision]);
 const choose=(ids:Set<string>)=>{setSelected(ids);localStorage.setItem(`touchline-selection-${job.id}`,JSON.stringify({signature:data?.signature,ids:[...ids]}))};
 const action=async(fn:()=>Promise<void>)=>{setBusy(true);setError('');try{await fn();await refresh()}catch(e:any){setError(e.message)}finally{setBusy(false)}};
 const build=()=>action(async()=>{await api(`/jobs/${job.id}/windows`,{run_missing_pose:missing});setData(d=>({...d,status:'working'}))});
 const working=busy||data?.status==='working'||exportData?.status==='working';
 const blocked=working||job.status==='processing'||job.identity_status==='scoring';
 const rows:Json[]=data?.windows||[];
 return <>
  <div className="page-heading compact"><div><span className="eyebrow">{view==='windows'?'STEP 5 · BUILD WINDOWS':'STEP 6 · DOWNLOAD DATASET'}</span><h1>{view==='windows'?'A dataset, one window at a time.':'Take your collection further.'}</h1><p>{job.target} · {job.match}</p></div></div>
  {(error||data?.error||exportData?.error)&&<div className="toast error" role="alert">{error||data?.error||exportData?.error}</div>}
  {job.imported_collection&&<div className="dataset-note">Existing processed collection imported. Tracking and cached poses are reused. Original review provenance is unspecified; inspect samples before treating them as human-confirmed data.</div>}
  {data?.stale&&<div className="toast error" role="alert">Review or pose data changed. Rebuild windows to refresh this collection. Preview and window export are disabled until then.</div>}
  {view==='windows'?<>
   <section className="surface dataset-controls"><div><h2>Build pose windows</h2><p>2 seconds · 1-second stride · 50 pose samples at 25 Hz</p><p>{data?.cached_poses??'—'} cached poses / {data?.accepted??'—'} accepted tracklets. Short or low-quality intervals are skipped with a reason.</p></div><div className="dataset-control-actions"><label className="check-label"><input type="checkbox" checked={missing} onChange={e=>setMissing(e.target.checked)} disabled={blocked}/>Estimate missing poses <small>runs the pose model</small></label><button className="primary" disabled={blocked||!data?.accepted} onClick={build}><RefreshCw size={16}/>{data?.status==='working'?`Building ${data.progress}/${data.total}…`:data?.signature?'Rebuild windows':'Build windows'}</button>{data?.status==='working'&&<button className="quiet" onClick={()=>action(async()=>{await api(`/jobs/${job.id}/windows/pause`,{})})}>Pause after tracklet</button>}</div></section>
   <div className="results-stats"><Stat value={rows.length} label="Ready windows"/><Stat value={selected.size} label="Selected"/><Stat value={data?.skipped?.length||0} label="Skipped intervals"/><Stat value={rows.filter(w=>w.review_source==='human').length} label="Human-confirmed identity"/></div>
   <div className="dataset-toolbar"><div className="button-row"><button className="secondary" disabled={blocked||data?.stale||!rows.length} onClick={()=>choose(new Set(rows.map(w=>w.id)))}>Select all</button><button className="quiet" onClick={()=>choose(new Set())}>Clear selection</button><button className="quiet" onClick={()=>setShowSkipped(v=>!v)}>{showSkipped?'Hide':'Show'} skipped reasons</button></div><button className="primary" disabled={blocked||data?.stale||!selected.size} onClick={()=>onNavigate('dataset')}>Download dataset<ArrowRight size={16}/></button></div>
   {showSkipped&&<section className="surface skipped-list"><h2>Skipped intervals</h2>{!(data?.skipped?.length)&&<p>No skipped intervals.</p>}{(data?.skipped||[]).map((s:Json,i:number)=><div key={i}><span>{fmt(s.start)}–{fmt(s.end)}</span><span>{s.reason}</span><small>{s.tracklet_id}</small></div>)}</section>}
   {!rows.length?<section className="surface empty-results"><Layers3 size={36}/><h2>{data?.status==='working'?'Preparing your windows…':'Build your first windows.'}</h2><p>{data?.accepted?'Use cached poses for a fast build. Missing-pose estimation is optional.':'Accept player appearances in Retrieve and review first.'}</p>{!data?.accepted&&<button className="secondary" onClick={()=>onNavigate('results')}>Review appearances<ArrowRight size={15}/></button>}</section>:<><div className="candidate-grid window-grid">{rows.slice(0,limit).map(w=><article key={w.id} className={`surface window-card ${selected.has(w.id)?'selected':''}`}><div className="window-card-top"><label className="check-label"><input type="checkbox" aria-label={`Select window at ${fmt(w.start)}`} checked={selected.has(w.id)} disabled={blocked||data?.stale} onChange={()=>{const next=new Set(selected);next.has(w.id)?next.delete(w.id):next.add(w.id);choose(next)}}/>{fmt(w.start)}–{fmt(w.end)}</label><button className="quiet" disabled={data?.stale||blocked} onClick={()=>setPreview(w)} aria-label={`Preview window at ${fmt(w.start)}`}><Play size={15}/></button></div>{w.representative&&<button className="window-thumb" disabled={data?.stale||blocked} onClick={()=>setPreview(w)}><img loading="lazy" src={'/media/'+w.representative} alt={`Player at ${fmt(w.start)}`}/><Play size={22}/></button>}<div className="window-info"><span className={'tag '+(w.review_source==='human'?'good':'')}>{w.review_source==='human'?'Human-confirmed':w.review_source==='automatic'?'Automatically accepted':'Imported / unspecified review'}</span><p>{w.source_observations} source pose observations → 50 samples</p><p>Lower-body keypoints: {Math.round(w.pose_quality*100)}% confident</p><small title={w.tracklet_id}>{w.tracklet_id}</small>{w.flags.length>0&&<p>{w.flags.join(' · ').replaceAll('_',' ')}</p>}</div></article>)}</div>{rows.length>limit&&<button className="secondary" onClick={()=>setLimit(n=>n+48)}>Show more ({rows.length-limit} remaining)</button>}</>}
  </>:<>
   <section className="surface dataset-download"><h2>Choose what to include</h2><div className="dataset-modes">{[['tracklets','Tracklets','Reviewed portions and frame-level boxes'],['windows','Windows','Selected pose arrays and window metadata'],['both','Both','Tracklets, selected windows and their relationships']].map(([id,title,description])=><button key={id} className={'dataset-mode '+(mode===id?'selected':'')} disabled={working} onClick={()=>setMode(id)}><strong>{mode===id&&<Check size={15}/>} {title}</strong><span>{description}</span></button>)}</div><p>{data?.accepted||0} accepted tracklets · {selected.size} selected windows</p><label className="check-label"><input type="checkbox" checked={video} onChange={e=>setVideo(e.target.checked)} disabled={working}/>Include player MP4 clips <small>decodes source footage; may take longer</small></label>{video&&<label className="field context-field">Source-frame context (seconds)<input type="number" min="0" max="10" step="1" value={context} onChange={e=>setContext(Number(e.target.value))}/><small>0 exports player clips only. Context clips are separate files.</small></label>}<div className="dataset-note">Every ZIP includes CSV/JSON manifests and a README. Pose windows include observed data, resampled arrays, confidence and validity masks. Original videos and model weights are excluded.</div><div className="card-footer"><button className="secondary" onClick={()=>onNavigate('windows')}>Choose windows</button><button className="primary" disabled={blocked||!data?.accepted||(mode!=='tracklets'&&(!selected.size||data?.stale))} onClick={()=>action(async()=>{await api(`/jobs/${job.id}/dataset`,{mode,window_ids:[...selected],include_video:video,context:video?context:0});setExportData(d=>({...d,status:'working'}))})}><Download size={16}/>{exportData?.status==='working'?`Preparing ${exportData.progress}/${exportData.total||'…'}…`:'Prepare dataset ZIP'}</button></div></section>
   {exportData?.stale&&<div className="toast error">Your previous ZIP is stale after a review or pose change. Prepare a new dataset.</div>}
   {exportData?.download_url&&!exportData.stale&&<section className="surface dataset-ready"><div><h2>Your dataset is ready.</h2><p>{exportData.mode} · {(exportData.bytes/1048576).toFixed(1)} MB · source-aligned metadata included</p></div><a className="primary" href={exportData.download_url} download><Download size={17}/>Download ZIP</a></section>}
   <section className="surface azure-exports" aria-label="Azure dataset exports">
    <div className="azure-heading"><div><h2><CloudUpload size={21}/>Save to Azure</h2><p>Keep a private cloud copy of your prepared dataset ZIP and download it later.</p></div><span className="tag">{cloud?.available?'Configured · private storage':'Setup required'}</span></div>
    <p className="azure-detail">Uploads the prepared ZIP, including any MP4 clips you chose. Each saved snapshot retains its metadata and review provenance. Processing stays local.</p>
    {cloud&&!cloud.available&&<p className="dataset-note">{cloud.message} <a href="https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/storage/azure-storage-blob" target="_blank" rel="noreferrer">About Azure Blob Storage</a></p>}
    {cloud?.available&&<p className="azure-detail">Destination: {cloud.account_url.replace('https://','')} / {cloud.container}</p>}
    {cloud?.error&&<div className="toast error" role="alert">{cloud.error}</div>}
    {cloud?.status==='working'&&<p role="status" aria-live="polite">Saving to Azure… {cloud.total>0?`${Math.round(100*cloud.progress/cloud.total)}%`:''}</p>}
    <button className="secondary" disabled={!cloud?.available||working||cloud?.status==='working'||!exportData?.download_url||exportData?.stale} onClick={()=>action(async()=>{await api(`/jobs/${job.id}/azure-exports`,{});setCloud(c=>({...c,status:'working',error:null}))})}><CloudUpload size={17}/>{cloud?.status==='working'?'Saving…':'Save prepared ZIP to Azure'}</button>
    {!exportData?.download_url&&<p className="azure-detail">Prepare a current dataset ZIP above before saving a new snapshot.</p>}
    {!!cloud?.exports?.length&&<div className="azure-history"><h3>Saved snapshots</h3>{cloud.exports.map((saved:Json)=><div className="azure-snapshot" key={`${saved.account_url}/${saved.container}/${saved.id}`}><div><strong>{new Date(saved.created*1000).toLocaleString()}</strong><p>{saved.tracklet_count} tracklets · {saved.window_count} windows · {(saved.bytes/1048576).toFixed(1)} MB{saved.included_video?' · includes MP4 clips':''}</p><small>SHA-256 {saved.sha256.slice(0,16)}… · {saved.historical?'Earlier review snapshot':'Matches current review'}</small></div>{saved.download_available?<a className="secondary" href={saved.download_url} download><Download size={16}/>Download from Azure</a>:<span className="azure-detail">Configure this snapshot’s Azure destination to download.</span>}</div>)}</div>}
   </section>
  </>}
  {preview&&<div className="modal-backdrop" onClick={()=>setPreview(null)}><section className="preview-modal surface" role="dialog" aria-modal="true" aria-label="Window preview" onClick={e=>e.stopPropagation()}><div className="panel-title"><h2>{fmt(preview.start)}–{fmt(preview.end)} · {job.target}</h2><button className="icon-button" aria-label="Close preview" onClick={()=>setPreview(null)}><X size={18}/></button></div><video controls autoPlay muted src={`/api/jobs/${job.id}/windows/${preview.id}/preview`}/><p className="dataset-note">Source-video preview. Pose arrays are resampled independently; the video retains source timing.</p></section></div>}
 </>;
}
function Stat({value,label}:{value:number,label:string}){return <div className="stat"><strong>{value}</strong><span>{label}</span></div>}
