# RLHF

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
