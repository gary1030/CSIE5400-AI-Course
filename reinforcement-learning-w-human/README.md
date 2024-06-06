# RLHF

## Environment

1. GPU: RTX 3080 Ti
2. GPU RAM: 12GB
3. Pytorch 2.1.0

## Installation

```sh
conda create -y -n ai_hw6 python=3.10
conda activate ai_hw6
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install --no-deps trl peft accelerate bitsandbytes
pip install tqdm packaging wandb
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
```

or

```sh
conda create -y -n ai_hw6 python=3.10
conda activate ai_hw6
pip install -r requirements.txt
```

## Run

```sh
# bash run.sh <exp_name> <model_name> <wandb_token> <optimizer> <epoch> <beta>
bash run.sh DPO unsloth/llama-3-8b-bnb-4bit <wandb_token> paged_adamw_32bit 1 0.1
bash run.sh ORPO unsloth/llama-3-8b-bnb-4bit <wandb_token> paged_adamw_32bit 1 0.1
```
