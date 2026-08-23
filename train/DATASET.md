# Dataset format

PixWorld trains on **posed multi-view scenes**: a set of frames from one clip, each with a
camera pose and intrinsics, plus a caption, plus optional depth.

The loader reads a plain local directory. There is no database, no object store and no
credential anywhere in the pipeline — one JSON Lines index and image folders.

## Layout

```
DATA_ROOT/
  index.jsonl                        one JSON object per scene
  scenes/
    <source>/<scene_id>/
      rgb/000000.jpg  000001.jpg  …  zero-padded; array order IS temporal order
      depth/000000.png …             optional, uint16 PNG
```

Frame ordinals replace timestamps, so temporal order is simply the array order in
`frames`. Nothing joins on filenames.

## One index line

| field | type | required | meaning |
|---|---|---|---|
| `scene` | string | yes | unique id across all sources |
| `source` | string | yes | a tag you choose; drives the mixture weights |
| `split` | `"train"` / `"val"` | no | if absent, derived from `md5(scene) % 20 == 0` |
| `text` | string | yes | the caption, **raw** — the trainer adds the prompt prefix |
| `root` | string | yes | scene directory, relative to `DATA_ROOT` |
| `frames` | list[string] | yes | RGB paths under `root`, in temporal order |
| `intrinsics` | list[[fx, fy, cx, cy]] | yes | per frame, **normalised by that frame's own pixel size** |
| `w2c` | list[list[float] × 12] | yes | per frame, row-major 3×4 world-to-camera, **OpenCV** |
| `depth` | list[string] or null | no | per-frame depth path; `""` for a frame with none |
| `depth_scale` | float | no | `metres = png_uint16 / 65535 * depth_scale` |
| `k_novel` | int or null | no | per-scene override of the novel-view count |

`len(frames) == len(intrinsics) == len(w2c)`, and `len(depth)` too when it is not null.
A scene is dropped if it has fewer than `views + k_novel` frames.

### Example

```json
{
  "scene": "0d8f65989586d136",
  "source": "re10k",
  "split": "train",
  "text": "a cozy indoor living room with a stone fireplace and a black sofa",
  "root": "scenes/re10k/0d8f65989586d136",
  "frames": ["rgb/000000.jpg", "rgb/000001.jpg"],
  "intrinsics": [[0.48979256, 0.87074226, 0.5, 0.5],
                 [0.48979256, 0.87074226, 0.5, 0.5]],
  "w2c": [[0.99603122, -0.01030123, -0.08840664, -0.10031936,
           0.00398936,  0.99744850, -0.07127769, -0.01486679,
           0.08891533,  0.07064211,  0.99353093, -0.09154028],
          [0.99574286, -0.01019396, -0.09160910, -0.10674299,
           0.00359057,  0.99740106, -0.07195991, -0.01481931,
           0.09210457,  0.07132463,  0.99319160, -0.09368711]],
  "depth": ["depth/000000.png", "depth/000001.png"],
  "depth_scale": 6.492793,
  "k_novel": null
}
```

## Conventions

**Poses are OpenCV world-to-camera.** x right, y down, +z forward. The loader inverts them
and converts to the OpenGL camera basis itself. Camera-to-world poses must be inverted
first; OpenGL poses (y up, −z forward) need the y and z **columns** of the rotation flipped
before inverting.

**Intrinsics are normalised by the source image size**, not the training resolution:
`fx = f_pixels / image_width`, `cy = c_pixels / image_height`. Store the images unmodified;
the loader handles the resize and crop.

**Do not pre-normalise the poses.** No re-rooting, no rescaling — the loader normalises
input and novel views jointly per sample, and divides depth by the same scale.

**Depth is z-depth** along the optical axis, in the same units as the pose translations.

## Building an index

```bash
# a per-scene JSON already carrying frames/intrinsics/w2c
python train/tools/build_index.py --root data/mine --format json

# RealEstate10K-style .txt camera files next to an image folder
python train/tools/build_index.py --root data/re10k --format re10k

# validate an existing index without writing one
python train/tools/build_index.py --root data/mine --check
```

## Source mixture

`source` groups scenes into pools; `SOURCE_WEIGHTS` sets how often each is drawn. Without
it, sampling is uniform over rows, so a pool ten times larger gets ten times the steps.

```bash
SOURCE_WEIGHTS='{"clean":0.5,"raw":0.2,"synthetic":0.3}' bash train/train_l2p.sh
```
