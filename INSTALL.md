# Installation

Verified on Python 3.10, PyTorch 2.6.0 + CUDA 12.4, A100-80GB.

```bash
conda create -n pixworld python=3.10 -y && conda activate pixworld
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r train/requirements.txt    # training (a superset of inference)
pip install -r infer/requirements.txt    # inference only
```

## gsplat

gsplat JIT-compiles its CUDA extension on the first render, not at import. Set your
architecture and warm it once:

```bash
export TORCH_CUDA_ARCH_LIST=8.0      # 8.0 A100 · 8.6 A6000/3090 · 8.9 L40S/4090 · 9.0 H100
python -c "from gsplat.cuda._backend import _C; print('gsplat ok')"
```

The training launchers do this before `torchrun`; the manual step is for interactive use.
If the build fails, `nvcc --version` usually disagrees with the CUDA your PyTorch was built
against.

## Weights

Wan2.2 — the backbone for training, the text encoder everywhere:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --local-dir weights/Wan2.2-TI2V-5B-Diffusers
```

PixWorld — for inference; skip it if you are training from Wan2.2 yourself:

```bash
python -c "from modelscope import snapshot_download; print(snapshot_download('SensenGao/PixWorld'))"
```

That directory holds `PixWorld-L2P-Wan5B/` and `PixWorld-L2P-Wan5B-4steps/`, each with a
`.safetensors` and the `config.json` that `--ckpt` reads beside it. Keep each pair together.

## Check it works

```bash
python train/selftest.py
```

Four checks — pose conventions and ray maps, the noise schedule, camera paths, and a gsplat
forward/backward. The last needs a GPU; each prints `... self-test OK`.

## If imports break

`numpy < 2` is required. Some vendor PyTorch images ship shim packages that hook into
`transformers` at import time and break `from_pretrained`:

```bash
pip uninstall -y transformer_engine transformer-engine transformer-engine-cu12 \
                 transformer-engine-torch bitsandbytes
```
