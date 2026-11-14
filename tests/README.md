# FairAge tests

From the repository root, run:

```text
python -m unittest discover -s tests -v
```

Most tests use small synthetic inputs. The [pretrained-weight check](test_model.py) skips if the [checkpoint](../README.md#download-pretrained-weights) is missing.

- [Data preparation](test_data_preparation.py) and [metrics](test_metrics.py)
- [Training](test_training.py), [resuming runs](test_recovery.py), and [Optuna](test_tune.py)
