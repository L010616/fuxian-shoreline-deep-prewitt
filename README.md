# Current reproduction archive (version 1.1.0)

The complete version 1.1.0 archive is publicly available at https://doi.org/10.5281/zenodo.23083486. It includes source, nine frozen neural checkpoints, six manually delineated reference shorelines and the completed index/JRC, expanded-scope and temporal analyses. The compact numerical cache and verification script independently reproduce 192 rows and all eight metrics. This verifies thresholds and evaluation, not training or inference; original provider imagery is required for those stages.

The file v1.1.0_source_and_weights.zip in this repository supplies the updated research source and weights. Download the complete Zenodo archive for numerical caches. CITATION.cff supplies the version-specific software citation. The original v1.0.0 archive remains available at https://doi.org/10.5281/zenodo.23050761. All methods and unfavourable comparisons are retained.

## Original repository description

# Fuxian Lake shoreline extraction with spatial priors and deep Prewitt gradients

This repository contains the code, Google Earth Engine export scripts, frozen data-split manifests, and numerical result tables supporting the manuscript **Fine shoreline extraction of Fuxian Lake using spatial-prior constraints and a deep Prewitt gradient branch**.

## Method overview

The workflow combines a compact DFANet semantic segmentation backbone with two complementary mechanisms:

1. a fixed spatial-prior channel derived from the 2019 JRC annual water extent and a 300 m candidate shoreline belt; and
2. a deep edge branch that expands fixed green/NIR Prewitt gradients through a residual depthwise-separable multiscale encoder before progressive semantic fusion.

The repository also contains the parameter-matched ordinary-convolution control, component ablations, final prediction-blind confirmation, cross-lake transfer evaluation, manual-shoreline evaluation, and lake-date hierarchical uncertainty analysis.

## Repository layout

```text
src/               Python model, training, evaluation, and statistical scripts
gee/               Google Earth Engine data selection and export scripts
manifests/         Frozen date splits and evaluation manifests
results/           Machine-readable tables and summary outputs used in the paper
environment/       Reproducible Python dependency specification
```

Raw Landsat 8 and Sentinel-2 GeoTIFFs are not redistributed. They can be recreated from the public Earth Engine collections with the scripts in `gee/`. The exported rasters must retain their native projection and the filenames recorded in the frozen manifests.

## Environment

The experiments were run with Python 3, PyTorch 2.11.0+cu128, NumPy 2.5.3, SciPy 1.18.1, Rasterio 1.5.1, scikit-image 0.26.0, and Pillow 12.3.0. Install the remaining pinned packages with:

```bash
pip install -r environment/requirements-lock.txt
```

Install a PyTorch build appropriate for the local CUDA or CPU environment separately.

## Data preparation

1. Run the Earth Engine inventory and export scripts in numerical order.
2. Download each exported Landsat 8 image, near-date Sentinel-2 image, and validity-mask GeoTIFF.
3. Preserve the exported filenames and arrange them according to the JSON/CSV files in `manifests/`.
4. Do not use Sentinel-2 imagery as a model input. It is used only to construct prediction-blind shoreline references within the common valid domain.

The manuscript uses 12 training dates, 3 validation dates, 6 pre-sealed primary blind-test dates, 6 edge-development diagnostic dates, and 6 final prediction-blind confirmation dates. Cross-lake transfer uses 18 nested dates from Dianchi Waihai, Erhai, and Xingyun Lake.

## Main reproduction entry points

- `src/run_multidate_ablation_pilot.py`: base DFANet and spatial-prior ablations.
- `src/run_edge_depth_validation_screen.py`: validation-only edge-depth selection.
- `src/run_parameter_matched_conv_control.py`: parameter-matched learned-convolution control.
- `src/run_prewitt_confirmation_set3_evaluation.py`: locked final six-date confirmation.
- `src/run_round4_final_blind_evaluation.py`: primary blind-test evaluation.
- `src/run_q2_cross_lake_zero_shot.py`: frozen-weight transfer to three unseen lakes.
- `src/run_q2_hierarchical_lake_statistics.py`: two-stage lake-date hierarchical bootstrap.
- `src/evaluate_q2_manual_core6_r2.py`: evaluation against manually delineated shorelines.
- `src/evaluate_q2_manual_core6_r2_uncertainty.py`: reference-shoreline perturbation sensitivity.

The locked evaluation scripts intentionally refuse to overwrite an existing formal result. Run them in a clean output directory when independently reproducing the analysis.

## Statistical conventions

The original Fuxian tables average per-seed metrics within each date; cross-lake and manual comparisons evaluate thresholded three-seed mean probabilities. Date-level summaries report means and between-date sample standard deviations. Exact six-date Wilcoxon results remain descriptive because dates can be dependent; the added confirmation sensitivity groups the six dates into three temporal episodes. Cross-lake and manual comparisons use lake-equal summaries and a two-stage lake-date hierarchical bootstrap. With only three lakes, the intervals describe consistency in the observed lakes and are not interpreted as population-wide significance. The post-review index/JRC comparison retains all dates and outcomes.

## Reproducibility scope

The files in `results/` are machine-readable outputs used to audit the manuscript tables and figures. Raw satellite raster inputs are omitted. The updated checkpoints are included in v1.1.0_source_and_weights.zip. Compact verification caches and the complete current numerical outputs are available in the published version 1.1.0 Zenodo archive. The older top-level source/results folders are retained for the earlier snapshot; use the version 1.1.0 package for the current analysis.

## License

Source code is released under the MIT License. Satellite products remain subject to the terms of their original providers.
