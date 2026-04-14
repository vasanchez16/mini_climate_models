# gridpoint_ml - Claude Code Build Prompt

Build a Python package called `gridpoint_ml` that trains ML models in parallel for every gridpoint in a spatiotemporal grid. Scaffold the full package structure and implement each module according to the following specification.

## Package Structure

```
gridpoint_ml/
├── config/
│   └── example_config.toml
├── gridpoint_ml/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── worker.py
│   ├── grid.py
│   ├── data.py
│   ├── metadata.py
│   └── io.py
├── user_scripts/
│   └── example_train.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_grid.py
│   ├── test_data.py
│   └── test_worker.py
├── pyproject.toml
└── run_pipeline.py
```

## Problem
Train one ML model per (time, lat, lon) gridpoint combination, totaling approximately 5,124 models.

## Parallelism
Use `concurrent.futures.ProcessPoolExecutor` with `max_workers` defined in the TOML config. Track progress with `tqdm` via `as_completed`.

## Training Data
- Features come from a single CSV where rows = simulations and columns = features. The CSV is implicitly ordered so row index maps directly to simulation number — there is no explicit sim ID column.
- Targets come from multiple NetCDF4 files, one per simulation, following a user-defined path pattern such as `/path/to/sim_{sim_id}/sim_{sim_id}_variable.nc`
- `sim_id` formatting is configurable via `sim_id_format` in the TOML (e.g. `"03d"` for zero-padded, `"d"` for plain integer)
- For each gridpoint, each worker iterates over all simulations, opens each simulation's NetCDF4 file independently (do not share file handles across processes), and slices the scalar value at `(time, lat, lon)` to construct `y` of shape `(n_simulations,)`
- `X` is shape `(n_simulations, n_features)` — the full features CSV

## User-Defined Training Script
Dynamically imported by the pipeline. Must implement:
- `train(X, y) -> model` — takes numpy arrays, returns a trained model
- `save(model, path)` — framework-native saving logic
- `load(path) -> model` — framework-native loading logic

## TOML Config
```toml
[grid]
times = ["2020-01-01 00:00", "2020-01-01 12:00"]
lats = [30.0, 31.0, 32.0]
lons = [-90.0, -89.0, -88.0]

[data]
features_csv = "path/to/features.csv"
target_netcdf_pattern = "/path/to/sim_{sim_id}/sim_{sim_id}_variable.nc"
sim_id_format = "03d"
n_simulations = 117
target_variable = "temperature"

[pipeline]
max_workers = 64
output_dir = "path/to/save/models"
training_script = "user_scripts/my_train.py"

[metadata]
save_metadata = true
```

## Output Per Gridpoint
- Model saved via the user-defined `save()` function
- JSON metadata file recording gridpoint coordinates, training script used, timestamp, and output path

## Additional Notes
- NetCDF4 file handles must be opened and closed inside each worker function — never passed from the parent process
- `example_train.py` in `user_scripts/` should demonstrate a simple sklearn model as a reference implementation
- The package should be installable via `pip install -e .`
- Implement all modules fully, not just stubs
