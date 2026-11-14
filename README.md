# FairAge

Does ImageNet pretraining help DeiT-Tiny classify apparent age? FairAge compares training from scratch, fine-tuning the full pretrained model, and training a new head on a frozen pretrained backbone.

## Results

The fixed FairFace splits contain 77,805 training, 8,645 development, and 10,954 holdout images. The holdout is the publisher’s validation split. The labels describe apparent age.

| Model | Development macro-F1 | Holdout macro-F1 | Holdout accuracy |
|---|---:|---:|---:|
| Baseline scratch | 0.445595 | 0.447877 | 0.482655 |
| Baseline pretrained | 0.529039 | 0.515123 | 0.538890 |
| Frozen pretrained | 0.312483 | 0.324170 | 0.401862 |
| Tuned scratch | 0.499418 | 0.486899 | 0.516250 |
| Tuned pretrained | 0.554338 | 0.527755 | 0.546558 |

These are FP32 evaluations of checkpoints selected using BF16 development macro-F1. The study used one seed, and the two Optuna searches used different search spaces and trial counts. See the [results notebook](notebooks/03_results_analysis.ipynb) for error and confidence comparisons.

## Install and use the environment

Python 3.12 or later is required. Run commands from the repository root.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebooks]"
```

**Linux and macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks]"
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python`. For the package alone, run `python -m pip install -e .`. List commands with `python -m fairage --help`.

## Notebooks

Open the [notebooks](notebooks/README.md) in VS Code or Jupyter, select the FairAge environment, and run the cells in order. The default workflows use public CSVs and a synthetic example on CPU. File exports are off by default.

If Jupyter needs a registered kernel, run:

```text
python -m ipykernel install --user --name fairage --display-name "FairAge"
```

## Reproduce training

Training requires FairFace images and an NVIDIA GPU with CUDA and BF16 support. Install [CUDA-enabled PyTorch and torchvision](https://pytorch.org/get-started/locally/).

### Prepare the data

Get the [padding=0.25 image archive](https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view?usp=sharing), [training labels](https://drive.google.com/file/d/1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH/view?usp=sharing), and [validation labels](https://drive.google.com/file/d/1wOdja-ezstMEp81tX1a-EYkFebev4h7D/view?usp=sharing) from [FairFace](https://github.com/joojs/fairface#data).

```text
data/raw/fairface-padding-0.25/
├── fairface_label_train.csv
├── fairface_label_val.csv
└── images/
    ├── train/
    └── val/
```

```text
python -m fairage.data prepare-splits --data-root data
```

This applies the study’s duplicate exclusions and seed-4531 split. It writes to `data/splits/fairface-v1/`, which must be a new directory.

### Download pretrained weights

For pretrained or frozen training, download the [pinned DeiT-Tiny weights](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k/resolve/80e968688553f219e4a86f940ed945a23709c16f/pytorch_model.bin) to `data/weights/deit_tiny_imagenet1k/pytorch_model.bin`. These are non-distilled ImageNet-1k weights from [timm](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k). FairAge keeps the new nine-class head.

### Run the baselines

```text
python -m fairage train scratch --run baseline_scratch
python -m fairage train pretrained --run baseline_pretrained
python -m fairage train frozen --run baseline_frozen
```

The baselines use [baseline.yaml](configs/baseline.yaml), seed 4531, and CUDA/BF16. To change paths, copy the YAML to a local file, edit `data_root`, `run_root`, or `weights`, and pass `--config PATH`.

The tuning campaign's launch documentation at commit `6e413de` records Optuna 5.0.0. This version is pinned, with multivariate TPE, no group decomposition, 24 expected-improvement candidates, and constant-liar behavior configured explicitly. The saved seed and startup-trial count still govern each study. Restarting preserves journal history and settings but does not restore sampler RNG state, so future suggestions may differ from an uninterrupted search; exact replay of the historical trials is not claimed.

## Project files

- [Source](src/fairage/README.md) and [configurations](configs/)
- [Published results](results/README.md)
- [Research paper](docs/deliverables/paper/)
- [Presentation](docs/deliverables/presentation/FairAge_Research_Presentation.pptx) and [figure assets](docs/deliverables/assets/README.md)
- [Tests](tests/README.md) · `python -m unittest discover -s tests -v`
