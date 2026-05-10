# gridpoint_ml — Prediction Pipeline Design

## Overview

This document describes the design for extending the `gridpoint_ml` package with a prediction pipeline. The prediction pipeline sources every trained gridpoint model to generate predictions for a new set of test features, then assembles the results into NetCDF4 output files organized by time point.

---

## User-Defined Training Script — Updated Contract

The training script now requires four functions. All four must be defined in the same file:

```python
def train(X, y):
    """
    X: np.ndarray of shape (n_train_samples, n_features)
    y: np.ndarray of shape (n_train_samples,)
    returns: trained model object
    """
    ...
    return model

def save(model, path):
    """
    Save the model in a framework-native format.
    """
    ...

def load(path):
    """
    Load and return the model from the given path.
    returns: model object
    """
    ...
    return model

def predict(model, X):
    """
    Run predictions on X and return a dictionary of named output arrays.
    Each value must be a np.ndarray of shape (n_variants,).
    Example for sklearn Gaussian Process:
    """
    mean, std = model.predict(X, return_std=True)
    return {
        "temperature_mean": mean,
        "temperature_std": std
    }
```

### Key design decisions:
- `predict()` returns a **dictionary** of named arrays to support multiple outputs (e.g. mean and std from a Gaussian Process)
- Each dictionary key becomes a separate variable in the output NetCDF4 file
- Variable names are therefore defined by the user in their `predict()` function, not in the TOML config
- All output arrays must be of shape `(n_variants,)`
- The pipeline validates that `train`, `save`, `load`, and `predict` are all present at import time

---

## Prediction TOML Configuration

A separate TOML file is used to configure each prediction run:

```toml
[prediction]
metadata_dir = "path/to/metadata/json/files"
test_features_csv = "path/to/test_features.csv"
output_dir = "path/to/save/netcdf4/files"
max_workers = 64
```

### Notes:
- `metadata_dir` points to the folder containing per-gridpoint JSON metadata files produced by the training pipeline
- `test_features_csv` follows the same format as the training features CSV — rows = variants, columns = features — but will typically have many more rows (potentially ~1,000,000)
- Variable names in the output NetCDF4 files come from the dictionary keys returned by the user's `predict()` function, not from the TOML
- `max_workers` controls the `ProcessPoolExecutor` worker count, same as the training pipeline

---

## Prediction Pipeline — Two Phase Design

### Phase 1 — Parallel Prediction

**Orchestration:**
- `concurrent.futures.ProcessPoolExecutor` with `max_workers` from the TOML
- Progress tracked with `tqdm` via `as_completed`
- Test features CSV is loaded once in the main process and passed to each worker as a numpy array

**Each worker:**
1. Reads its assigned gridpoint metadata JSON to retrieve:
   - Model file path
   - Path to the training script used
2. Dynamically imports the training script using `importlib` (same mechanism as training pipeline)
3. Loads the trained model via `load(path)`
4. Runs `predict(model, X_test)` → returns dictionary of arrays each of shape `(n_variants,)`
5. Saves result as a `.npz` temp file to `{output_dir}/tmp/` named by gridpoint coordinates

**Example temp file naming:**
```
{output_dir}/tmp/pred_t2017-08-01T10_30_lat1.8750_lon-0.9375.npz
```

**Failure handling:**
- If a worker fails for any reason, log a warning identifying the gridpoint
- Write a `.npz` temp file filled with `NaN` values of the same shape `(n_variants,)` for each expected output variable
- Pipeline continues with remaining gridpoints
- NaN values propagate into the final NetCDF4 output, clearly marking failed gridpoints

---

### Phase 2 — Assembly

Runs sequentially after all Phase 1 workers complete.

**Steps:**
1. Scan `{output_dir}/tmp/` for all `.npz` temp files
2. Group temp files by time point
3. For each unique time point:
   - Load all gridpoint `.npz` files belonging to that time point
   - For each variable (dictionary key), assemble a 2D array of shape `(n_lats, n_lons, n_variants)` by placing each gridpoint's `(n_variants,)` array at its corresponding `(lat, lon)` position
   - Write a NetCDF4 file with dimensions `(lat, lon, variant)` where `variant` is an integer index `0, 1, 2, ... n_variants-1`
   - Each variable from the predict dictionary becomes a separate variable in the NetCDF4 file
4. Delete `{output_dir}/tmp/` after all NetCDF4 files are successfully written

**Output file naming convention:**
```
predictions_YYYY_MM_DD_HH_MM_SS.nc
```

**Example output files** for a grid with 2 timestamps per day × 61 days = 122 files:
```
predictions_2017_08_01_10_30_00.nc
predictions_2017_08_01_22_30_00.nc
predictions_2017_08_02_10_30_00.nc
...
```

---

## Output NetCDF4 File Structure

Each output file corresponds to one time point and contains:

| Property | Value |
|---|---|
| Dimensions | `(lat, lon, variant)` |
| `lat` | float, gridpoint latitudes |
| `lon` | float, gridpoint longitudes |
| `variant` | integer index 0 to n_variants-1 |
| Variables | one per key in `predict()` return dict |

**Example for a Gaussian Process predict function:**
```
predictions_2017_08_01_10_30_00.nc
├── dimensions: lat, lon, variant
├── variable: temperature_mean (lat, lon, variant)
└── variable: temperature_std  (lat, lon, variant)
```

---

## Temporary File Format

- Format: `.npz` (NumPy compressed archive)
- One file per gridpoint
- Each `.npz` contains one named array per output variable, each of shape `(n_variants,)`
- `.npz` was chosen over `.npy` because:
  - Naturally supports multiple named arrays in one file
  - Built-in compression reduces I/O overhead at large variant counts (~1,000,000)
  - Halves the number of temp files compared to one `.npy` per variable per gridpoint

---

## Memory Considerations

At ~1,000,000 variants × 5,124 gridpoints × 8 bytes (float64):
- **Total prediction data: ~41 GB**
- This rules out returning all predictions to the main process — each worker writes directly to disk
- Temporary disk space required: ~41 GB × number of output variables (e.g. ~82 GB for mean + std)
- This space is reclaimed after successful assembly

---

## Updated Package Structure

```
gridpoint_ml/
├── config/
│   ├── example_config.toml              # training config
│   └── example_predict_config.toml      # prediction config
├── gridpoint_ml/
│   ├── __init__.py
│   ├── pipeline.py                      # training orchestration
│   ├── worker.py                        # training worker
│   ├── grid.py                          # grid parsing
│   ├── data.py                          # feature/target loading
│   ├── metadata.py                      # per-gridpoint JSON metadata
│   ├── io.py                            # model save/load
│   ├── predict_pipeline.py              # prediction orchestration
│   ├── predict_worker.py                # prediction worker
│   └── assembler.py                     # NetCDF4 assembly (Phase 2)
├── user_scripts/
│   └── example_train.py                 # example with all four functions
├── tests/
│   ├── test_pipeline.py
│   ├── test_grid.py
│   ├── test_data.py
│   ├── test_worker.py
│   ├── test_predict_pipeline.py
│   └── test_assembler.py
├── pyproject.toml
├── run_pipeline.py                      # training CLI entrypoint
└── run_predictions.py                   # prediction CLI entrypoint
```

---

## New Modules

### `predict_pipeline.py`
- Reads prediction TOML config
- Loads test features CSV once in main process
- Scans metadata directory for all gridpoint JSON files
- Orchestrates `ProcessPoolExecutor` with tqdm
- Calls Phase 2 assembly after all workers complete

### `predict_worker.py`
- Receives gridpoint metadata and test features
- Dynamically imports training script via `importlib`
- Loads model, runs `predict()`, saves `.npz` temp file
- Handles failures gracefully with NaN fill and warning log

### `assembler.py`
- Groups `.npz` temp files by time point
- Assembles and writes per-time NetCDF4 output files
- Cleans up `{output_dir}/tmp/` after successful assembly

---

## CLI Entrypoint

```bash
python run_predictions.py predict_config.toml
python run_predictions.py predict_config.toml --verbose
```

Follows the same pattern as `run_pipeline.py` with argparse and logging configuration.
