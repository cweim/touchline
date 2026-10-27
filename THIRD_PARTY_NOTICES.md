# Third-party notices

## Application source

A top-level open-source license for Touchline's original application code has not
been selected. The notices below identify third-party components; they do not
assign one license to the entire repository.

## Vendored source

- `retrieval/vendor/osnet.py` is adapted from
  [Kaiyang Zhou's Torchreid / deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).
  Its MIT license and copyright notice are retained in
  [TORCHREID_LICENSE](retrieval/vendor/TORCHREID_LICENSE).

## Dependencies and separately downloaded models

| Component | Source and licensing reference | Distribution in this repository |
| --- | --- | --- |
| Ultralytics YOLO and ByteTrack implementation | [Ultralytics source and license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE); the pinned Python package reports AGPL-3.0 | Dependency declaration only; no package source or weights bundled |
| Football player detector checkpoint | Download location follows [Roboflow sports soccer setup](https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh) | URL and reference hash only; a public download URL is not a separate license grant for the checkpoint |
| OSNet MSMT17 checkpoint | [Torchreid model zoo](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md) | URL and reference hash only; checkpoint/training-data conditions must be checked separately |
| Optional DINOv2 baseline | [DINOv2 source and Apache-2.0 license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | Setup instructions only; source and weights are downloaded into ignored `models/` |
| YOLOv8 pose checkpoint | [Ultralytics assets](https://github.com/ultralytics/assets), with Ultralytics licensing references above | Setup instructions only; weights are ignored |
| Azure Identity and Azure Blob Storage SDKs | [Azure SDK for Python](https://github.com/Azure/azure-sdk-for-python), MIT | Optional dependencies only |

Other installed Python and npm packages retain their own licenses and notices.
Check the exact versions installed for your intended distribution. FFmpeg is a
separately installed system prerequisite; this repository does not bundle it.

## Demo media and input footage

The approved demo and poster are in `docs/media/`. Their music, sound effects and
production credits are in [Demo credits](docs/media/CREDITS.md).

The recording includes third-party football broadcast imagery. A cleared public
redistribution permission for that imagery is not recorded in this repository.
The application's source, screenshots and music credits do not confer broadcast
rights. Match videos, runtime collections, derived dataset exports and model
checkpoints are not distributed as sample data.
