# GraphWave Distillation

Implementation for **GraphWave Distillation for Continual Molecular Property
Prediction** (NeurIPS 2025).

## Repository Status

Released. All code required to reproduce the continual-learning experiments is
included: graph wavelet transforms, the spectral distillation objective,
training scripts and evaluation on the twelve-assay stream.

## Layout

```
src/
  wavelets.py    graph wavelet transform
  distill.py     spectral distillation objective
  model.py       property prediction model
  train.py       continual training loop
eval/
  stream_eval.py evaluation over the assay stream
experiments/
  neurips_main.yaml
results/
  stream_results.json
```

## Citation

```bibtex
@inproceedings{scott2025graphwave,
  title={GraphWave Distillation for Continual Molecular Property Prediction},
  author={Scott, Jeffrey and Ibrahim, Samir},
  booktitle={NeurIPS},
  year={2025}
}
```
