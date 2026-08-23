"""Writing results out: images, videos, and 3D Gaussian files."""
import os

import numpy as np
import torch

from geometry.render import prune_opacity

__all__ = ["save_image", "save_grid", "save_video", "save_npz", "save_ply",
           "sh0_to_rgb"]

SH_C0 = 0.28209479177387814


def sh0_to_rgb(sh0):
    """Degree-0 spherical harmonics -> the RGB a viewer shows for a flat-lit splat."""
    return (sh0 * SH_C0 + 0.5).clip(0.0, 1.0)


def save_image(t, path):
    """``t`` ``[3, H, W]`` in ``[-1, 1]`` -> PNG."""
    from PIL import Image
    arr = ((t.float().clamp(-1, 1).permute(1, 2, 0).cpu() + 1) * 127.5).round().clamp(0, 255)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(arr.byte().numpy()).save(path)


def save_grid(views, path, cols=4, pad=4, bg=16):
    """``views`` ``[V, 3, H, W]`` in ``[-1, 1]`` -> one contact sheet."""
    from PIL import Image
    V, _, H, W = views.shape
    rows = (V + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W + (cols + 1) * pad, rows * H + (rows + 1) * pad),
                      (bg, bg, bg))
    for i in range(V):
        arr = ((views[i].float().clamp(-1, 1).permute(1, 2, 0).cpu() + 1) * 127.5)
        im = Image.fromarray(arr.round().clamp(0, 255).byte().numpy())
        r, c = divmod(i, cols)
        sheet.paste(im, (pad + c * (W + pad), pad + r * (H + pad)))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sheet.save(path)
    return sheet.size


def save_video(frames, path, fps=12):
    """``frames`` ``[T, 3, H, W]`` in ``[-1, 1]`` -> mp4.

    Tries imageio, then the ``ffmpeg`` binary, then a GIF.
    """
    arr = ((frames.float().clamp(-1, 1).permute(0, 2, 3, 1).cpu() + 1) * 127.5)
    arr = arr.round().clamp(0, 255).byte().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None
    if imageio is not None:
        try:
            imageio.mimsave(path, list(arr), fps=fps, quality=8, macro_block_size=1)
            return path
        except Exception:                                        # imageio without ffmpeg
            pass
    if _ffmpeg_write(arr, path, fps):
        return path
    if imageio is not None:
        path = os.path.splitext(path)[0] + ".gif"
        imageio.mimsave(path, list(arr), duration=1.0 / fps)
        return path
    raise RuntimeError(
        f"cannot encode {path}: install `imageio[ffmpeg]` or put `ffmpeg` on PATH")


def _ffmpeg_write(arr, path, fps):
    """Pipe ``[T, H, W, 3]`` uint8 straight into ffmpeg.  True if a file came out."""
    import shutil
    import subprocess
    exe = shutil.which("ffmpeg")
    if exe is None:
        return False
    t, h, w, _ = arr.shape
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           # yuv420p + even dimensions is what makes the file play outside ffplay;
           # 480x832 is already even, but a custom --height/--width need not be.
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path]
    try:
        p = subprocess.run(cmd, input=arr.tobytes(), capture_output=True)
    except OSError:
        return False
    return p.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0


def save_npz(scene, cameras, path, prune=0.01):
    """Write the raw Gaussian tensor plus its cameras, for reloading in Python."""
    g = prune_opacity(scene.detach().float(), prune).cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, gaussians=g.astype(np.float16),
                        cameras=cameras.detach().float().cpu().numpy(),
                        layout="xyz3|opacity1|scale3|rotation4|sh27", sh_degree=2)
    return g.shape[0]


def save_ply(scene, path, prune=0.01, max_points=0):
    """Write a 3D Gaussian Splatting ``.ply`` that standard splat viewers can open.

    The interchange format expects **unactivated** parameters -- the viewer applies
    ``exp`` to the scales and ``sigmoid`` to the opacity itself -- while the renderer here
    works with activated ones.  So this inverts both.

    ``max_points`` optionally keeps only the most opaque N, which matters because 16 views
    at 480x832 with 2 Gaussians per pixel is 12.8M points -- around 3 GB of ASCII-free
    binary PLY, more than most viewers will open.
    """
    g = prune_opacity(scene.detach().float(), prune).cpu()
    if max_points and g.shape[0] > max_points:
        keep = torch.topk(g[:, 3], max_points).indices
        g = g[keep]
    xyz = g[:, 0:3]
    opacity = g[:, 3:4].clamp(1e-6, 1 - 1e-6)
    scales = g[:, 4:7].clamp(min=1e-9)
    rot = g[:, 7:11]
    sh = g[:, 11:38]                                    # 9 coefficients x 3 channels

    opacity_logit = torch.log(opacity / (1 - opacity))  # inverse sigmoid
    scales_log = torch.log(scales)                      # inverse exp
    # The renderer stores SH coefficient-major (9, 3); the PLY format wants the DC term
    # first and then the rest channel-major.
    shr = sh.reshape(-1, 9, 3)
    f_dc = shr[:, 0, :]
    f_rest = shr[:, 1:, :].permute(0, 2, 1).reshape(-1, 24)

    fields = (["x", "y", "z", "nx", "ny", "nz"]
              + [f"f_dc_{i}" for i in range(3)]
              + [f"f_rest_{i}" for i in range(24)]
              + ["opacity"] + [f"scale_{i}" for i in range(3)]
              + [f"rot_{i}" for i in range(4)])
    data = torch.cat([xyz, torch.zeros_like(xyz), f_dc, f_rest,
                      opacity_logit, scales_log, rot], dim=1).numpy().astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {data.shape[0]}\n".encode())
        for name in fields:
            f.write(f"property float {name}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())
    return data.shape[0]
