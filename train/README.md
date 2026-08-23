# Training PixWorld

Two stages. Stage 1 fine-tunes Wan2.2-TI2V-5B on 16 posed views into a pixel-space
multi-view 3D generator; stage 2 distils that model to four steps with DMD2.

| | script | steps | starts from | produces |
|---|---|---|---|---|
| Stage 1 | `train_l2p.py` / `train_l2p.sh` | 50,000 | Wan2.2-TI2V-5B | the 50-step model |
| Stage 2 | `train_dmd2.py` / `train_dmd2.sh` | 10,000 | a stage-1 checkpoint | the 4-step model |

Stage 2 is optional: the stage-1 model is usable on its own.

## Quick start

```bash
pip install -r requirements.txt

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --local-dir weights/Wan2.2-TI2V-5B-Diffusers

# your dataset — see DATASET.md
python train/tools/build_index.py --root data/mine --format json
python train/tools/build_index.py --root data/mine --check

DATA_ROOT=data/mine bash train/train_l2p.sh
TEACHER=runs/l2p/model_step50000.pt DATA_ROOT=data/mine bash train/train_dmd2.sh
```

Multi-node is standard torchrun:

```bash
NNODES=4 NODE_RANK=$i MASTER_ADDR=<host0> DATA_ROOT=data/mine bash train/train_l2p.sh
```

Every hyper-parameter is an environment variable in the launcher, carrying the default used
for the released models — read `train_l2p.sh` / `train_dmd2.sh` for the full list, or
`python train/train_l2p.py --help`. `MAX_STEPS=100000 LR=3e-5 bash train/train_l2p.sh` does
what it looks like. Both launchers auto-resume from the newest checkpoint in `OUT_DIR`.

The geometry perception loss is off by default. `GEO_LOSS=pi3` or `GEO_LOSS=vggt` turns it
on; it needs [Pi3](https://github.com/yyfz/Pi3) or
[VGGT](https://github.com/facebookresearch/vggt) cloned and its weights fetched, via
`GEO_REPO` and `GEO_WEIGHTS`. The released checkpoints were not trained with it.

## Publishing a checkpoint

```bash
python train/tools/convert_checkpoint.py runs/l2p/model_step50000.pt \
    PixWorld-L2P-Wan5B/PixWorld-L2P-Wan5B.safetensors --safetensors \
    --ema runs/l2p/model_ema_step50000.pt
python train/tools/convert_checkpoint.py runs/dmd2/model_step10000.pt \
    PixWorld-L2P-Wan5B-4steps/PixWorld-L2P-Wan5B-4steps.safetensors --safetensors --distilled
```

This writes a `config.json` beside the weights, which `infer/infer.py` reads; keep the two
together. The released checkpoints are on
[ModelScope](https://www.modelscope.cn/models/SensenGao/PixWorld).

## Self-tests

```bash
python train/selftest.py             # all four
python train/selftest.py cameras     # just one
```

`cameras`, `schedule` and `trajectories` run on CPU; `render` needs a GPU. Each prints
`... self-test OK`.
