# Inference

Text or a single image goes in; an explorable 3D Gaussian scene comes out.

```bash
pip install -r infer/requirements.txt

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --local-dir weights/Wan2.2-TI2V-5B-Diffusers    # for the text encoder

# weights: https://www.modelscope.cn/models/SensenGao/PixWorld
export PIXWORLD=$(python -c "from modelscope import snapshot_download; print(snapshot_download('SensenGao/PixWorld'))")
export FEW=$PIXWORLD/PixWorld-L2P-Wan5B-4steps/PixWorld-L2P-Wan5B-4steps.safetensors
export BASE=$PIXWORLD/PixWorld-L2P-Wan5B/PixWorld-L2P-Wan5B.safetensors
export WAN=weights/Wan2.2-TI2V-5B-Diffusers
```

Each `.safetensors` sits beside a `config.json` that `--ckpt` reads, so keep the two
together. One script drives both checkpoints; the config says which it is.

## Three things it does

**Text → 3D** — a prompt and a camera path:

```bash
python infer/infer.py --ckpt $FEW --wan_path $WAN \
    --cameras infer/examples/poses/t2mv_ridge.json \
    --prompt "a rocky mountain ridge at sunset above the sea" \
    --out out/ridge
```

**Image → 3D** — the image is view 0 of the path and stays pinned there:

```bash
python infer/infer.py --ckpt $FEW --wan_path $WAN \
    --cameras infer/examples/poses/i2mv_bedroom.json \
    --image infer/examples/i2mv_bedroom/reference.jpg \
    --prompt "a bedroom with light walls and a black metal bed" \
    --out out/bedroom
```

**Reconstruction** — 16 posed views in, one forward pass at σ = 0. No sampler, no guidance,
no prompt:

```bash
python infer/infer.py --ckpt $BASE --wan_path $WAN --mode recon \
    --cameras infer/examples/recon_bedroom/cameras.json \
    --views_dir infer/examples/recon_bedroom/views \
    --out out/recon_bedroom
```

## Output

```
views/v00.png … v15.png    the generated views
grid.png                   all views as one contact sheet
sweep.mp4                  the camera path rendered from the Gaussian field
scene.ply                  the Gaussians — opens in any 3DGS viewer
scene.npz                  the same, as raw arrays
meta.json                  prompt, seed, gaussian count
```

`--video_frames 81` sets the video length by resampling the 16-camera path;
`--video_round_trip` flies it out and back so the clip loops. `--video_frames 0` skips it.

`scene.ply` stores **unactivated** parameters — viewers apply `exp` to the scales and
`sigmoid` to the opacity. 16 views at 480×832 is 12.8M points, so `--ply_max_points` keeps
the most opaque 2M by default.

## Cameras

`--cameras` takes a JSON of 16 cameras, `[ qw qx qy qz | tx ty tz | fx fy cx cy ]`,
quaternion real-first, `fx`/`cx` normalised by image width and `fy`/`cy` by height. Frame 0
is the identity pose and translations are scaled so the largest is ≈ 1; a file already in
that form passes through untouched, anything else is normalised on load. Write your own with
`save_cameras_json` in `geometry/cameras.py`.

`--trajectory {dolly,orbit,pan,spiral}` synthesises a path instead; `--data_root` with
`--scene` reuses one from a dataset index.

Everything else: `python infer/infer.py --help`. `infer/examples/` has six ready-to-run
cases, and `infer/make_examples.py` runs them all through both checkpoints in one go.
