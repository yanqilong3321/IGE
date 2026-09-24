# Amazon-Book data

The main experiment uses `train.pkl`, `valid.pkl`, and `test.pkl` under this directory. Each file is a SciPy CSR interaction matrix with the same user/item shape. The evaluator consumes `train.pkl` and `test.pkl`; `valid.pkl` is retained for provenance and compatibility with the upstream release.

Do not add Gowalla, Yelp, MovieLens, Tmall, historical runs, or analysis artifacts to this release directory.
