# AGENTS.md

Guidance for AI coding agents and new contributors working in this repository.

## What this is

Qwen-3D is a generalist 3D vision-language model (ECCV 2026) built on Qwen2.5-VL with a
detectron2 / Mask2Former-style training stack. `train.py` is the training and evaluation
entry point and is driven by `scripts/main_qwen.sh`. README.md, docs/RUN.md and
docs/DATA.md are the user-facing documentation; keep them accurate when you change
behaviour or dependencies.

## Layout

- `qwen3d/` - the model (`model.py`, `modeling_qwen2_5_vl_modified.py`, `modeling/`),
  the data pipeline (`data_video/`), the yacs config (`config.py`, `configs/*.yaml`) and
  utilities. Importing `qwen3d` registers all datasets (it prints `Registering ...`).
- `libs/pointops2/` - CUDA point-cloud ops from Pointcept, compiled into the environment
  (`libs/pointops` and `libs/pointgroup_ops` are vendored but unused by the training code).
- `data_preparation/`, `tools/` - dataset preprocessing and precomputation scripts.
- `scripts/` - launchers (`setup.sh` defines `configure_local` / `configure_slurm`,
  `main_qwen.sh` builds the `train.py` command) and `check_env.py`.
- `docs/` - RUN.md (commands), DATA.md (datasets and paths), `init.sh` (manual setup helper).
- `splits/` - dataset split files.

## Environment

The environment is defined by the `[tool.pixi.*]` tables in `pyproject.toml` and pinned by
`pixi.lock`. The manual conda recipe in README.md is kept for users without pixi; keep the
two in sync when dependencies change.

- `pixi install` creates `.pixi/envs/default` (about 13 GB) and compiles `libs/pointops2`.
- `pixi run setup` downloads the NLTK stopwords corpus (one time).
- `pixi run check` is the smoke test (CUDA, compiled ops, key imports). Run it after any
  change to the environment or to `libs/`.
- `pixi shell` or `pixi run <cmd>` runs commands in the environment. Activation sets
  `PYTHONPATH` to the repository root and `CUDA_HOME` to the environment prefix, so run
  commands from the repository root.
- `pixi run train -- <config overrides>` wraps `scripts/main_qwen.sh`; environment variables
  such as `BS=1 EVAL_ONLY=1` go before `pixi run`.

Rules for dependencies and builds:

- Use only the toolchain inside the environment: nvcc 12.9, the CUDA headers and libraries
  and GCC 14 come from conda-forge, and `CC`, `CXX`, `CUDA_HOME`, `CUDACXX` and
  `CUDAHOSTCXX` are preset. Never point a build at `/usr/local/cuda*` or the system compiler.
- Prefer conda-forge packages (`pixi add <package>`). Use `[tool.pixi.pypi-dependencies]`
  only for packages conda-forge does not have, and avoid URL dependencies on GitHub release
  assets (use the conda package, e.g. `spacy-model-en_core_web_sm`, or a pip-based task).
- Keep the exact pins: PyTorch 2.12.1 (cuda129 builds), transformers 4.51.1, tokenizers,
  accelerate, peft, timm, datasets and NumPy 1.26. `qwen3d/modeling_qwen2_5_vl_modified.py`
  imports private functions from transformers, so bumping transformers requires re-porting
  that file.
- Commit `pixi.lock` together with every manifest change. `.pixi/` is gitignored.
- `libs/pointops2` is a pypi path dependency built without build isolation against the
  conda PyTorch. After editing its sources, force a rebuild with
  `pixi reinstall pointops2`, then run `pixi run check`.

## Running and verifying

- There is no unit-test suite. Verification means `pixi run check`, `python train.py --help`
  (imports everything and registers the datasets) and, when data is available, the commands
  in docs/RUN.md.
- Training and evaluation need the datasets and paths described in docs/DATA.md.
  `scripts/main_qwen.sh` hard-codes cluster paths (`DETECTRON2_DATASETS`, `*_DATA_DIR`,
  `FEATURE_DIR`, `OUTPUT_DIR_PREFIX`) that must be adapted to the machine.
- Configuration is detectron2 / yacs: defaults live in `qwen3d/config.py`, the experiment
  config is `qwen3d/configs/qwen_3d.yaml`, and overrides are passed as `KEY VALUE` pairs at
  the end of the command line (see docs/RUN.md for the dataset combinations).
- Pretrained weights are loaded with `MODEL.WEIGHTS` and must match `QWEN_MODEL`
  (3B or 7B backbone); checkpoints go in `ckpts/` and outputs in `output*/`, both gitignored.

## Conventions

- Python 3.12, PyTorch 2.12.1 and the detectron2 API; match the surrounding code style. The
  code base is not uniformly formatted, so do not reformat files you are not otherwise changing.
- Do not commit data, checkpoints, logs or visualizations; `.gitignore` already covers the
  usual locations.
- Document user-visible changes in README.md or the relevant file in `docs/`.
