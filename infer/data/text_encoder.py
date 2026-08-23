"""Text conditioning.

Wan conditions on UMT5 embeddings.  PixWorld uses the *same* encoder and the same
tokenisation, so a prompt means the same thing to the pretrained transformer blocks as it
did during Wan's own training.

The multi-view branch prefixes every caption with :data:`MV_PREFIX`.
"""
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

try:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
except ImportError:                                                # pragma: no cover
    import re
    import html

    def prompt_clean(text):
        text = html.unescape(html.unescape(str(text)))
        return re.sub(r"\s+", " ", text).strip()

__all__ = ["DEFAULT_NEG", "MV_PREFIX", "TextEncoder", "DEFAULT_PROMPTS"]

#: Prefix added to every multi-view caption.
MV_PREFIX = "[Static] "

#: The default negative prompt for the multi-view sampler.  It is Wan2.2's own default
#: negative, which is why it is Chinese.
DEFAULT_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)

#: A few prompts for text-to-3D sampling during training and in the demo.
DEFAULT_PROMPTS = [
    "a cozy living room with a stone fireplace, a leather sofa and warm lamp light",
    "a sunlit modern kitchen with white cabinets, a marble island and a window over the sink",
    "an empty art gallery with white walls, wooden floors and skylights",
    "a quiet suburban street lined with trees on a clear autumn morning",
    "a hotel lobby with a curved staircase, marble floor and a large chandelier",
]


class TextEncoder:
    """UMT5 prompt encoder.

    Args:
        model_path: a Wan diffusers checkpoint directory; the ``tokenizer`` and
            ``text_encoder`` subfolders are used.
        max_len: sequence length.  226 is Wan's.

    Call it with a list of strings to get ``[B, max_len, 4096]``.

    Note the padding: the encoder is run with the real attention mask, then each sequence
    is truncated to its true length and zero-padded back to ``max_len``.  Encoding with
    the mask and *then* padding is not the same as letting the model attend over pad
    tokens, and the transformer was trained on the former.
    """

    def __init__(self, model_path, device, dtype=torch.bfloat16, max_len=226):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
        self.text_encoder.requires_grad_(False)
        self.device, self.dtype, self.max_len = device, dtype, max_len

    @torch.no_grad()
    def __call__(self, prompts):
        prompts = [prompt_clean(p) for p in prompts]
        ti = self.tokenizer(prompts, padding="max_length", max_length=self.max_len,
                            truncation=True, add_special_tokens=True,
                            return_attention_mask=True, return_tensors="pt")
        ids, mask = ti.input_ids.to(self.device), ti.attention_mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = self.text_encoder(ids, mask).last_hidden_state.to(self.dtype)
        emb = [u[:v] for u, v in zip(emb, seq_lens)]
        return torch.stack(
            [torch.cat([u, u.new_zeros(self.max_len - u.size(0), u.size(1))])
             for u in emb], dim=0)

    def free(self):
        """Drop the encoder from GPU memory.

        Inference embeds every prompt first and then loads the transformer, which is what
        makes a 5B model plus a 5B text encoder fit on one card.
        """
        self.text_encoder = self.text_encoder.to("cpu")
        del self.text_encoder
        self.text_encoder = None
        torch.cuda.empty_cache()
