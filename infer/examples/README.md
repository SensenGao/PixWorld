# Examples

Six cases. **Inputs only** — run them and look at your own outputs.

```
inputs.json                     every case, machine-readable
poses/*.json                    16-camera paths
i2mv_*/reference.jpg            the image-to-3D reference frames
recon_*/cameras.json            16 input cameras + 16 held-out cameras
recon_*/views/v00..v15.jpg      the 16 posed views to reconstruct from
```

| case | task | scene |
|---|---|---|
| `t2mv_living_room` | text → 3D | indoor · room |
| `t2mv_ridge` | text → 3D | outdoor · mountain ridge at sunset |
| `i2mv_bedroom` | image → 3D | indoor · room |
| `i2mv_mountains` | image → 3D | outdoor · mountains and a fjord |
| `recon_bedroom` | reconstruction | indoor · room |
| `recon_coast` | reconstruction | outdoor · coastline |

The camera paths and the reconstruction views come from held-out validation clips of
RealEstate10K and ACID.

## Running them

All six, loading each checkpoint once:

```bash
python infer/make_examples.py \
    --ckpt_multi $BASE --ckpt_few $FEW --wan_path $WAN \
    --examples infer/examples --out out/examples
```

One at a time: see [`../README.md`](../README.md).
