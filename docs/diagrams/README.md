# Touchline data pipeline

![Discover appearances, find the target player and package reusable data](touchline-pipeline-v2.png)

- [V2 PNG](touchline-pipeline-v2.png): 3840 × 2860, recommended for the submission's media section.
- [V2 SVG](touchline-pipeline-v2.svg): vector output with editable text.
- [Mermaid source](touchline-pipeline.mmd): compact logical flow for reuse.
- V1 remains available as [PNG](touchline-pipeline.png) and [SVG](touchline-pipeline.svg).

V2 groups the pipeline into three visual stages: discover appearances, find the
target player, and package reusable data. Filled player illustrations and filmstrips
show the data changing form; overlapping timeline intervals explain windows.
Reference-profile and candidate-vector arrows both enter the comparison step.
The direct tracklet-export route bypasses window building.

## Read the diagram

Tracking first creates candidate tracklets for the players visible in each camera
shot. It does not establish the target player's identity across different shots.
Clean reference appearances guide embedding-based retrieval; human review then
accepts, rejects or trims the ranked candidates.

Reviewed tracklets can be exported directly. Window building is a separate route
to export. The current window builder uses cached or explicitly estimated poses,
quality checks and gap limits. Downstream uses shown in the dashed panel are
examples of work enabled by the exported data, not implemented analytics claims.

All mini-frames and player figures are schematic illustrations drawn for this
diagram; no match footage, source crops or personal/cloud identifiers are used.

## Implementation details behind the labels

- Camera segmentation: FFmpeg scene-change detection.
- Wide shots: sampled YOLO detections plus heuristic duration, player-count,
  player-height and green-pitch checks. Defaults require at least 2 seconds,
  median player count of at least 6, median height below 25% of frame height and
  median green-pitch fraction of at least 32%; inconsistent sample views are rejected.
- Tracking: YOLO detections and ByteTrack reset within each camera shot.
- Crops: quality-based selection, with timestamps and original-frame boxes retained.
- Retrieval: OSNet or DINOv2 crop embeddings; ranking combines a reference-trained
  classifier score and cosine similarity, followed by quality aggregation/flags.
  Ranking scores are not calibrated identity probabilities.
- Review: retain accepted intervals and distinguish human, automatic and imported
  decision provenance.
- Windows: 2-second duration, 1-second stride; preserve reviewed/shot boundaries;
  the current pose sampler rejects gaps longer than 0.4 seconds.
- Export: tracklets, selected windows or both; local ZIP or private Azure Blob copy.

The figure was checked against `retrieval/pipeline.py`, `retrieval/vision.py`,
`retrieval/identity.py`, `retrieval/windows.py`, `retrieval/dataset.py` and
`retrieval/azure_export.py` on 2026-10-27.
