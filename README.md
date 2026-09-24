# When to Trust Intent Graphs? Evidence-Aware Gating for Cold-User Recommendation

This repository is the official code implementation for **When to Trust Intent Graphs? Evidence-Aware Gating for Cold-User Recommendation**. It provides the Amazon-Book main experiment for DCCF, BIGCF, LightGCN, and LightGCN + NT-SSM.

## Environment

Create an environment and install the dependencies listed in [`requirements.txt`](requirements.txt):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch installation, install the PyTorch build matching the local CUDA driver before running the command above. The reference environment uses PyTorch 2.5.1 with CUDA 12.1.

## Dataset

The demonstration dataset is **Amazon-Book**. The repository already contains the processed files under `data/amazon/`:

```text
data/amazon/train.pkl
data/amazon/valid.pkl
data/amazon/test.pkl
```

Each file is a SciPy CSR user-item interaction matrix with the same shape. The files are the public preprocessed Amazon-Book split released with [DCCF](https://github.com/HKUDS/DCCF). If the files are not downloaded with the repository, obtain them with:

```bash
git clone --depth 1 https://github.com/HKUDS/DCCF.git /tmp/DCCF
mkdir -p data/amazon
cp /tmp/DCCF/data/amazon/{train,valid,test}.pkl data/amazon/
```

Before evaluation, the code applies the following processing steps in `src/protocol.py`:

1. Load the three pickled matrices as CSR matrices and convert all nonzero values to implicit-feedback value 1.
2. Use `train.pkl` as the support-interaction source and `test.pkl` as the query-interaction source; assert that the two matrices do not overlap.
3. Randomly partition users with interactions in both matrices using split seed `20260915`: 20% test cold users and 10% calibration users, with the calibration users divided equally into fit and tune groups.
4. Remove fit, tune, and test users from warm-user graph training.
5. At evaluation, sample `k=5` support interactions per cold user, mask those support items from ranking, and rank the complete item catalog.

The main metrics are NDCG@20 and Recall@20. `valid.pkl` is retained for compatibility with the public split; this release protocol uses `train.pkl` and `test.pkl`.

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
numbers are under `metrics` in those files. The `runs/` directory is ignored by Git;
average the three seed values for the paper's main table.
