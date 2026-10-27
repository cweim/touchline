# GitHub preparation check — 2026-10-27

## Passed

- Copied only Git's non-ignored source/documentation/media list to an independent
  directory, without local settings, datasets, research artifacts or model weights.
- Created a fresh Python 3.11.8 virtual environment with no system site packages.
- Installed the declared application dependencies and test/Azure/import extras.
- `python -m pip check`: no broken requirements.
- `python -m pytest -q`: **57 passed**. One existing Starlette/AnyIO deprecation
  warning remains; no test failures.
- `npm ci`: successful locked dependency install; npm reported zero vulnerabilities.
- `npm run build`: successful TypeScript/Vite production build, Node.js 22.20.0.
- Fresh FastAPI lifespan and HTTP checks: frontend served, jobs/video lists empty,
  private-file access rejected, Azure routes available.
- Standalone CLI status returned an empty job list, using only the copied project.
- Built a Python wheel and confirmed it contains the Torchreid license notice;
  it contains no videos, model weights or runtime database.
- Checked public text files for common private-key/GitHub/Hugging Face/Azure secret
  patterns and personal absolute paths: no findings. This is a pattern scan, not
  a guarantee against every possible secret format.
- Verified local README links, file-size limits and exclusion of private folders.
- Copied the approved demo and poster unchanged; source/copy SHA-256 hashes match.
  Video: H.264/AAC, 1920×1080, 30 fps, 105 seconds, 11,397,447 bytes.

## Boundaries and remaining publication details

- New-video model inference was not rerun as part of the clean-install check.
  Models/input footage are supplied separately; tests exercise isolated fixtures.
- Azure save/download was confirmed by the developer in their configured setup.
  Clean-install cloud tests use a mocked storage boundary and do not access that
  account or verify it independently.
- The approved demo predates Azure export; the README describes that feature
  separately. The approved video has not been modified.
- A license for original Touchline application code has not been selected.
  Third-party notices and the vendored MIT notice are included.
- The demo/poster include third-party broadcast imagery; public redistribution
  permission has not been confirmed. Music and production credits are included.
- No GitHub remote, public upload or initial commit was created during this check.

The clean upload copy excludes runtime databases, source datasets, checkpoints,
cloud credentials, local settings, dependencies, render workspaces and internal
submission drafts. Publish only that copy or the intended non-ignored Git contents.
