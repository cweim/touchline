# Publishing Touchline

Publish the standalone `touchline` repository, not its parent research project.
The completed checks are recorded in [Release check](RELEASE_CHECK.md).
Use Git's tracked/non-ignored file list rather than uploading the entire working
directory through a browser or Finder. The working directory contains ignored
private runtime data and local settings.

## Public contents

- FastAPI/React source, dependency manifests and lockfile.
- Model setup/import scripts and reference checkpoint hashes.
- Isolated tests, README and third-party notices.
- `docs/media/touchline-demo.mp4`: the approved 105-second V2 demo.
- `docs/media/demo-poster.jpg` and `docs/media/CREDITS.md`.

## Exclusions

Source videos, downloaded weights, datasets, runtime databases, local settings,
credentials, node_modules, compiled frontend output, render projects and internal
submission planning files remain ignored. The demo MP4 is the sole video exception.
Never use `git add --force` on these private folders.

## README video

The README thumbnail links to the checked-in demo MP4 through a relative path.
This works without knowing the future GitHub owner/repository name. It provides
a clickable video link rather than relying on an inline HTML video player.

The file is approximately 11 MB, below GitHub's 100 MiB regular Git file limit.
GitHub's free-plan video attachment limit is 10 MB, so drag-and-drop attachments
are a different constraint. If replacing the video often, use a GitHub Release
asset or a video host rather than accumulating binary revisions in source history.

For the hackathon, also upload to an accepted host such as YouTube or Vimeo and
use that public URL in the submission form. A GitHub MP4 alone does not replace
the event's required video-host URL. The existing video is unchanged; Azure's
storage feature can be described in the written submission.

## Before pushing

1. Check the exact intended files with `git status --short` and
   `git ls-files --cached --others --exclude-standard`.
2. Review [third-party notices](../THIRD_PARTY_NOTICES.md), the license choice for
   original application code and permission to publicly distribute embedded
   broadcast imagery. Media credits document known sources, not rights clearance.
3. Run the README setup in an independent directory. A fresh app starts empty;
   supply your own permitted video and separately downloaded models for inference.
4. Run `python -m pytest -q` and `npm run build --prefix frontend`.
5. Stage and inspect the intended files, commit, then push to the existing
   `origin/main` at `https://github.com/cweim/touchline.git`. Never upload a ZIP
   of the populated working folder: it contains ignored data.

An upload-only clean copy is also prepared under the ignored local directory
`runtime/releases/touchline-github-ready/`. Its adjacent ZIP is a transport copy.
If using GitHub's browser uploader, upload the clean folder's contents, not the
ZIP itself. Continue development in the main project and regenerate this snapshot
after changes.

GitHub documentation: [large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
[attachments](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files).


## Connected-project presentation — October 28, 2026

The README links to [Movement Lens](https://github.com/cweim/movement-lens), shows
the shared data-flow diagram and points to the [dataset handoff guide](HANDOFF.md).
Touchline is the general dataset-preparation project; Movement Lens is a separate
downstream movement-exploration app. The apps exchange immutable exports rather
than sharing a live runtime or database.

The README screenshots are selected from the existing saved-session Touchline
walkthrough. Its match differs from Movement Lens's prepared example. Do not
present their counts as one archive transferred on screen.

Use these GitHub **About → Edit** fields:

- **Description:** Turn football broadcast video into reviewed player tracklets and traceable pose-window datasets for downstream analytics, with optional private Azure export.
- **Topics:** `football`, `sports-analytics`, `computer-vision`, `player-tracking`, `pose-estimation`, `dataset`, `azure-blob-storage`, `fastapi`, `react`.
- **Website:** add a real public hosted demo or accepted demo-video URL when available.

Repository metadata is separate from Git-tracked files. A README commit does not
set the About description, topics or website. Movement Lens's matching fields
are in its [publishing notes](https://github.com/cweim/movement-lens/blob/main/docs/PUBLISHING.md).


## Push this presentation update

From the `touchline` repository root:

```bash
git add README.md docs .github
git diff --cached --stat
git commit -m "docs: connect Touchline and Movement Lens with a visual handoff"
git push origin main
```

The local checkout now follows the existing public history; no force push or
new repository initialization is needed.
