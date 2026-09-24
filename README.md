# Evidence-Aware Gating: Amazon-Book Main Experiment

This directory is the clean release package for the paper's **main Amazon-Book experiment**. It contains only the training, prototype-adaptation, and cold-user evaluation code for the four reported backbone configurations:

- DCCF
- BIGCF
- LightGCN
- LightGCN + NT-SSM

Historical logs, result snapshots, ablation scripts, signal diagnostics, tuning experiments, audit reports, and other datasets are intentionally excluded.

## Environment

The reference environment is Python 3.10+, PyTorch 2.5.1 with CUDA 12.1, SciPy, NumPy, scikit-learn, and `torch-sparse`. A CUDA GPU is required by the released training/evaluation scripts.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the PyTorch build matching the CUDA driver. This is the reference build:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If the pre-built `torch-sparse` wheel is unavailable for the selected PyTorch/CUDA
combination, install the matching wheel first and then install the remaining packages:

```bash
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install 'numpy>=1.24' 'scipy>=1.10' 'scikit-learn>=1.3'
```

Check the installation before launching a long run:

```bash
python - <<'PY'
import torch, scipy, sklearn
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('scipy', scipy.__version__, 'sklearn', sklearn.__version__)
assert torch.cuda.is_available(), 'A CUDA GPU is required for this release'
PY
```

Set `CUDA_VISIBLE_DEVICES` when selecting a GPU, for example
`export CUDA_VISIBLE_DEVICES=0`.

## Dataset and protocol

The included `data/amazon/` files are the Amazon-Book interaction matrices used by the paper. The evaluator uses the shared support-only protocol:

- warm-user backbone training;
- frozen item embeddings at cold-user inference;
- support size `k=5` for the main table;
- full-catalog ranking with support items masked;
- fit/tune selection before test evaluation;
- split seed `20260915`.

The data files are public interaction matrices. Their checksums are recorded in `data/amazon/SHA256SUMS`.

The data are included in this repository so that the commands below work immediately.
If a mirror is preferred, replace the three files under `data/amazon/` with files having
the same names and verify them before running:

```bash
cd data/amazon
sha256sum -c SHA256SUMS
cd ../..
```

The expected files are `train.pkl`, `valid.pkl`, and `test.pkl`; the released evaluator
uses `train.pkl` and `test.pkl`, while `valid.pkl` is retained for provenance.

## Code layout

`src/train_eval.py` trains DCCF or BIGCF on warm users. `src/train_light_baseline.py` trains LightGCN, and `src/train_nt_baseline.py` trains the NT-SSM LightGCN variant. `src/prototype_adapter.py` constructs the offline K-means prototype interface for LightGCN-based encoders. `src/formal_eval.py` implements the shared cold-user scoring and evaluation code.

All runtime outputs must be written outside the source tree or to an ignored `runs/` directory. The repository does not contain paper result logs.

## Main experiment commands

All commands below are run from the repository root. The first command makes the local
modules importable and creates an ignored output directory:

```bash
export PYTHONPATH="$PWD/src"
mkdir -p runs
```

### DCCF and BIGCF

Train a warm-user encoder, then evaluate the frozen checkpoint with the common cold-user
evaluator. The evaluator performs fit/tune selection before producing the test metrics.
Run the following block once for each `MODEL` in `dccf bigcf` and each seed in `0 1 2`.

```bash
export PYTHONPATH="$PWD/src"

MODEL=dccf
SEED=0
NAME=${MODEL}_amazon_s${SEED}

python src/train_eval.py --model "$MODEL" --dataset amazon \
  --name "$NAME" --out "runs/$NAME" \
  --epochs 100 --batch 10240 --n_batches 40 --seed "$SEED" --train_only

python src/formal_eval.py --model "$MODEL" --dataset amazon \
  --checkpoint "runs/$NAME/${NAME}_checkpoint.pt" \
  --seed "$SEED" --k 5 --out "runs/${NAME}_eval" \
  --evidence_agg cosmax
```

For BIGCF, change only `MODEL=bigcf`. To obtain the three-seed main table, repeat the
same two commands with `SEED=1` and `SEED=2`, using a fresh output name each time.

### LightGCN

The LightGCN script saves a warm-user checkpoint. Pass it to the prototype adapter to
construct the offline K-means prototypes and run the same Amazon-Book cold-user evaluation.
Repeat this block for seeds `0`, `1`, and `2`.

```bash
SEED=0
NAME=lightgcn_amazon_s${SEED}
python src/train_light_baseline.py --dataset amazon --out "runs/$NAME" --seed "$SEED"
python src/prototype_adapter.py --model lightgcn --dataset amazon --seed "$SEED" \
  --checkpoint "runs/$NAME/best.pt" --out "runs/${NAME}_eval"
```

### LightGCN + NT-SSM

This is the NT-SSM loss/encoder applied to the same warm-user graph protocol. Repeat this
block for seeds `0`, `1`, and `2`.

```bash
SEED=0
NAME=ntssm_amazon_s${SEED}
python src/train_nt_baseline.py --dataset amazon --out "runs/$NAME" --seed "$SEED"
python src/prototype_adapter.py --model lightgcn_nt_ssm --dataset amazon --seed "$SEED" \
  --checkpoint "runs/$NAME/best.pt" --out "runs/${NAME}_eval"
```

The three training/evaluation blocks produce one `result.json` per seed. The principal
numbers are under `metrics` in those files. The `runs/` directory is ignored by Git and
is not part of this release. The repository intentionally does not include a result
aggregation script or historical logs; average the three seed values for the paper table.

For a short smoke run before a full training job, use one epoch and a small sample count:

```bash
python src/train_eval.py --model dccf --dataset amazon --name smoke --out runs/smoke \
  --epochs 1 --batch 256 --n_batches 1 --seed 0 --train_only
```

## Reproducibility note

The reported DCCF/BIGCF numbers are cold-user adaptations of frozen warm-user encoders, not native cold-start results reported by the original backbone papers. The benchmark split was explored during method development; it should be described as development/reproducibility evidence rather than an untouched confirmatory test.

## Upstream code

The minimal upstream model files are retained under `src/upstream/` with their original model names. Please consult the corresponding upstream repositories for their licenses and full training implementations before redistributing modified versions.
