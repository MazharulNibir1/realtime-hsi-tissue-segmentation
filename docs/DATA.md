# Data: HeiPorSPECTRAL (subset)

**Dataset.** HeiPorSPECTRAL — Heidelberg Porcine HyperSPECTRAL Imaging Dataset of 20 physiological
organs (Studier-Fischer et al., *Scientific Data* 10, 414 (2023), doi:10.1038/s41597-023-02315-8).
Project page: https://spectralverse-heidelberg.org/HeiPorSPECTRAL · Zenodo: doi:10.5281/zenodo.7737674

**Licence.** CC BY 4.0. Redistribution and adaptation are allowed with attribution and an
indication of changes. The demo sample crops in this repo are therefore redistributed under
CC BY 4.0 with a citation to the paper. (The proprietary TIVITA parameter-index formulas are not
used here.)

## Facts verified from the archive index

These numbers come from the remote zip's central directory (read via HTTP range requests;
no data was downloaded):

| | |
|---|---|
| Full archive | `HeiPorSPECTRAL.zip`, 907,375,987,596 bytes (845 GiB), ZIP64, 181,796 entries, server supports byte ranges |
| Camera | TIVITA Tissue, push-broom, 480 × 640 px, 100 bands, 500–1000 nm (~5 nm) |
| Cube file | `data/subjects/PXXX/<ts>/<ts>_SpecCube.dat`: float32 reflectance, (480, 640, 100), 122.9 MB raw, ~108 MB deflated in zip |
| Cubes | 5,756 organ images (+2 colour-checker cubes, ignored) |
| Subjects | 11 pigs (P086–P096) |
| Labels | 20 organs; **each image has exactly one organ annotated** (2 images have 2) as a polygon, by 3 annotators (`annotations/<ts>#polygon#annotatorN#<label>#binary.png`). All other pixels are unlabelled. |
| Per organ | 8 pigs × 36 images (4 situs × 3 angles × 3 repetitions); 287–288 images per organ |
| Preprocessed cubes | `intermediates/preprocessing/L1/<image>.blosc`: float16, (480, 640, 100), ~36 MB each |

Label indices (from `data/dataset_settings.json`): stomach 0, small_bowel 1, colon 2, liver 3,
gallbladder 4, pancreas 5, kidney 6, spleen 7, bladder 8, omentum 9, lung 10, heart 11,
cartilage 12, bone 13, skin 14, muscle 15, peritoneum 16, major_vein 17,
kidney_with_Gerotas_fascia 18, bile_fluid 19, unlabeled 255.

## Subset used

- **Cubes:** the dataset's preprocessed cubes (`intermediates/preprocessing/L1`). Each pixel
  spectrum is L1-normalised (sums to 1); a downloaded cube was checked to confirm this. All
  later normalisation options start from these L1 spectra.
- **Images:** one per situs for every (pig, organ) pair, preferring angle 0 and repetition 1,
  up to 4 per pair: **640 cubes**, exactly 4 per pair, 32 per organ. Two of them use the
  fallback angle (−25°). The exact list is in `configs/subset_heiporspectral.json`.
- **Download:** `scripts/download_data.py` (`make data`) fetches only the selected archive
  members with HTTP range requests (22.78 GB). The download resumes after interruption, and
  every file is checked against the zip CRC32 (`make data-verify`).
- **Safe loading:** `.blosc` files carry a small pickle header. It is read with an unpickler
  that only allows `numpy.dtype` (`src/hsi/data/blosc_io.py`), so no arbitrary code can run.
- **Working resolution:** cubes are binned 2×2 to 240 × 320 × 100 (float16, ~9.8 GB) in the
  pipeline (T3) and re-normalised after binning. 240 × 320 is close to the native resolution
  of a 16-band 4×4 snapshot mosaic sensor.
- **Labels:** binary polygon masks from all 3 annotators (KB each). Annotator 1 is used by
  default; a majority-vote option is available.
- **Split (by pig):** test = P095, P096; val = P093, P094; train = the other 7 pigs.
  Every organ appears in test and val, and each organ appears in at least 4 training pigs.

## Consequences of sparse labels

There is no background class, and only one organ is labelled per frame. All models are trained
with unlabelled pixels ignored, and all metrics (F1, Dice) are computed on annotated pixels only.
The demo overlay shows predictions for every pixel, but only annotated pixels were ever
supervised or evaluated.
