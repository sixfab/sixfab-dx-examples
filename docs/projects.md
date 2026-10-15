# Standalone DEEPX projects

`python_examples/` holds two complete, standalone apps with their own interactive `install.sh`, in addition to the single-file demos under `object_detection/`, `ppu/`, etc. Neither needs Frigate.

| Project | What it does |
|---|---|
| [python_examples/retail-analytics](../python_examples/retail-analytics) | People counting and a heatmap on the Pi's own screen |
| [python_examples/pose-estimation](../python_examples/pose-estimation) | Pilates rep counter, fall detection, and a pose-controlled game |

## Install

1. Install the `sixfab-dx` package (see the repo root [README](../README.md)) — this gives you the shared Python environment at `/opt/sixfab-dx/venv`.
2. `cd python_examples/<project> && ./install.sh`.

Both installers accept `--with-demo`, `--yes`, and `--dry-run`, and `VENV_DIR=<path>` to use your own Python environment instead of the shared one.

## Release assets

Each project ships its model (and optional demo video) as GitHub Release assets tagged `<project>-vX.Y.Z`. The installer reads `python_examples/<project>/models.txt` for the tag, file names, and SHA-256 values, and verifies each download. Set `RELEASE_REPO=<org>/<repo>` to use a fork.

## Licensing

See [LICENSE-NOTES.md](../LICENSE-NOTES.md) in the repo root for both models' licenses (Apache-2.0 and AGPL-3.0) and [DEEPX-SDK-LICENSE.txt](../DEEPX-SDK-LICENSE.txt) for the DEEPX SDK terms — both models are dxcom-compiled.

## Tests

```bash
cd python_examples
python -m pytest tests -q
bash tests/shell/run.sh
```

## Camera-based (Frigate) projects moved out

The five camera/Frigate apps (boss-detector, porch-pirate, bird-species-id, squirrel-sprinkler, smart-doorbell) now live in the sibling [sixfab-dx-frigate](https://github.com/sixfab/sixfab-dx-frigate) repo.
