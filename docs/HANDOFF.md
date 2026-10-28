# From Touchline to Movement Lens

[Touchline](https://github.com/cweim/touchline) prepares reusable player datasets from broadcast video. [Movement Lens](https://github.com/cweim/movement-lens) is one downstream application: inspect movement groups, find their observed moments across a match and share an explanation with its evidence.

![Touchline exports player-window data; Movement Lens checks, groups, reviews and explains it](diagrams/touchline-movement-lens.svg)

## Two independent applications

| | Touchline | Movement Lens |
| --- | --- | --- |
| Problem | Finding a player's usable appearances in broadcast footage | Making an extracted movement library easier to inspect and discuss |
| Input | A match video, half or shorter clip you may use | A compatible Touchline ZIP with pose windows |
| Main work | Wide-shot discovery, tracking, appearance retrieval and interval review | Quality checks, numerical descriptors, exploratory clustering and a recurrence timeline |
| Output | Tracklets and/or pose-window dataset with source provenance | A selected-cluster evidence pack with complete membership and representative clips |
| Microsoft integration | Optional private Azure Blob dataset storage | Microsoft Foundry explanation; optional Azure snapshot import and report storage |

The apps share a file contract, not a live database. Movement Lens does not need Touchline to be running. Touchline's exports can also serve other downstream applications.

## 1. Prepare the dataset in Touchline

1. Choose a video and target player; find wide shots and collect clean player references.
2. Retrieve candidate tracklets, then review or trim their retained intervals.
3. Build windows with source-bound pose data. If required poses are missing, explicitly enable pose estimation; previously imported legacy caches are not automatically verified.
4. In **Download dataset**, choose **Windows** or **Both**. Include MP4 clips for source-video/pose comparison; arrays-only input still supports numerical analysis. A tracklets-only export cannot supply movement windows.
5. Download the ZIP, or save the prepared snapshot to your private Azure Blob container.

For two source halves, use the same player and base match label. Supported labels such as `Match name · first source half` and `Match name · second source half` preserve half identity while grouping the same match. Keep source-relative timestamps and hashes unchanged.

The ZIP includes:

```text
manifest.json       # schema, player/match, source hash/FPS, keypoint order
tracklets.json      # reviewed parent intervals and provenance
tracklets.csv
windows.json        # two-second bounds, array/media references, quality metadata
windows.csv
windows/*.npz       # sampled and observed poses, confidence, validity and timing
windows/*.mp4       # optional player clips
```

Windows contain fifty resampled COCO-17 pose samples at twenty-five Hz. The samples are not fifty independent source observations; original observations and interpolation masks remain in the arrays.

## 2. Import into Movement Lens

Start Movement Lens using its [local setup](https://github.com/cweim/movement-lens#run-locally). Import through its local API; there is no cross-app upload wizard in the current three-screen interface.

```bash
curl -fsS -F 'file=@/absolute/path/to/first-half.zip' \
  http://127.0.0.1:8770/api/datasets

# Use the job_id returned above; wait until status is "done".
curl -fsS http://127.0.0.1:8770/api/jobs/JOB_ID
```

Repeat for the second half, then reload the app. A fresh installation chooses the largest available same-player match from the latest usable source snapshots. The private prepared demo pins its audited match separately. `GET /api/datasets` reports imported and usable counts; local `/docs` describes all request fields.

For a private Azure handoff, configure `MOVEMENT_LENS_AZURE_SOURCE_ACCOUNT_URL` and `MOVEMENT_LENS_AZURE_SOURCE_CONTAINER`. `GET /api/azure/datasets` lists snapshots, and `POST /api/azure/datasets/import` imports a selected snapshot with its SHA-256 and ETag. The download is checked before the same local importer runs. This is an explicit snapshot transfer, not automatic synchronization.

## 3. Understand which windows can be analysed

A valid archive may still contain withheld windows. Movement Lens checks source binding, original pose observations, required-joint visibility, source gaps, pose validity and review exclusions. A source hash alone does not prove the pose was computed from that source or certify the player's identity. Imported or uncertain review provenance stays visible.

An import with zero usable windows does not mean the ZIP disappeared. Check its per-window quality reasons through the dataset API. Re-exporting an old cache does not repair its provenance; source-bound preparation or a corrected upstream export is needed. The optional `scripts/verify_export.py` in Movement Lens can prepare a new source-bound snapshot from the exact original video and an existing pose checkpoint. It preserves the original input.

Overlapping windows are not extra exposure. Fitting uses independent observations, and observed duration merges overlapping source intervals. Missing broadcast visibility remains unknown.

## 4. Follow the evidence

Open a cluster → inspect typical and less-typical examples → confirm a descriptive category → find its observed intervals on the match timeline → explain the aggregate → export the evidence pack.

The local demonstration analyses Marcelo's Real Sociedad match across both halves: 290 usable independent windows, 580 observed seconds and two suggested groups. This is an example result from private prepared inputs, not bundled sample data or a promised result for every video. The older Touchline demo uses a different saved match; the videos illustrate the shared workflow rather than showing the same ZIP transferred on screen.

Foundry reads aggregate statistics, bounded sampled example frames, the timeline and primary research. General recovery context is distinct from measured player needs. The workflow does not establish fatigue, muscle loading, injury risk or treatment. The exported report keeps the complete cluster member list and original source timestamps beside the explanation.
