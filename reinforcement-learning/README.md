# Homework 5

## Install Necessary Packages

```sh
conda create -n hw5 python=3.11 -y
conda activate hw5
pip install -r requirements.txt
```

## Training

```sh
python pacman.py
```

## Evaluation

```sh
python pacman.py --eval --eval_model_path=submissions/pacma_dqn.pt
```
