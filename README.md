# CellSAM Pathology (Local GUI)

This repository packages a pathology GUI workflow on top of CellSAM, plus Windows OpenSlide binaries for WSI (SVS/NDPI/TIFF) support.

## What is included
- `gui_final.py`: main GUI entrypoint (human-in-the-loop labeling + analysis)
- `wsi_handler.py`: OpenSlide-based WSI reader
- `preprocessing_pipeline.py`: nuclei detection + feature extraction pipeline
- `report_engine.py`: HTML report generation
- `cellsam/`: vendored CellSAM library and docs
- `openslide_bin/`: Windows OpenSlide binaries and licenses

## Requirements
- Python 3.10+
- GPU optional (uses CUDA if available)
- OpenSlide DLLs are bundled under `openslide_bin/`

## Setup
1) Create and activate a virtual environment.
2) Install dependencies. The base CellSAM dependencies live in `cellsam/requirements.txt`:

```bash
python -m venv .venv
.\venv\Scripts\activate
pip install -r cellsam/requirements.txt
```

3) Configure OpenSlide (Windows only):

```bash
python setup_openslide.py
```

4) If you want CellSAM model download, set the token:

```powershell
$env:DEEPCELL_ACCESS_TOKEN=\"<your_token_here>\"
```

## Run the GUI

```bash
python gui_final.py
```

## Tests / Diagnostics
- `smoke_test.py`
- `test_cellsam_on_wsi.py`
- `test_tissue_detection.py`
- `test_real_svs.py`

## Notes
- The GUI creates a local SQLite file named `<slide>_data.db` in the working directory.
- Output reports are written to `reports/` by default.
