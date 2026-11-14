# FairAge source

- [Data preparation](data.py) creates the FairFace splits and loads images.
- [DeiT-Tiny](model/deit.py) defines the model. [Weight loading](model/weights.py) keeps the new age head.
- [Training](train.py) saves and resumes runs. [Tuning](tune.py) runs the Optuna studies.
- [Evaluation](evaluate.py) uses [shared metrics](metrics.py). [Analysis](analysis.py) and [plots](plots.py) read saved results.
- [CLI](cli.py) handles the commands in the [project guide](../../README.md).
