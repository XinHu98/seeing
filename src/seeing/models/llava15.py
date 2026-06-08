"""LLaVA-1.5 helpers for CODA-LM experiments."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any

import torch
from PIL import Image


@dataclass
class LlavaBundle:
    tokenizer: Any
    model: Any
    image_processor: Any
    context_len: int
    constants: Any
    conv_templates: Any
    mm_utils: Any


def _import_llava():
    try:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import tokenizer_image_token, process_images
        from llava.utils import disable_torch_init
    except ImportError as exc:
        raise ImportError(
            "LLaVA is required for this command. Install the official LLaVA package "
            "or put its repository on PYTHONPATH before running CODA-LM LLaVA-1.5 scripts."
        ) from exc
    constants = type("LlavaConstants", (), {
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
    })
    mm_utils = type("LlavaMMUtils", (), {
        "tokenizer_image_token": tokenizer_image_token,
        "process_images": process_images,
    })
    return constants, conv_templates, load_pretrained_model, mm_utils, disable_torch_init


def load_llava15(
    model_path: str = "liuhaotian/llava-v1.5-7b",
    model_name: str = "llava-v1.5-7b",
    device: str | torch.device = "cuda",
    device_map: str | dict | None = None,
    attn_implementation: str = "eager",
) -> LlavaBundle:
    constants, conv_templates, load_pretrained_model, mm_utils, disable_torch_init = _import_llava()
    disable_torch_init()
    kwargs = {"attn_implementation": attn_implementation}
    if device_map is not None:
        kwargs["device_map"] = device_map
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, **kwargs)
    if device_map is None:
        model = model.to(device)
    model.eval()
    return LlavaBundle(tokenizer, model, image_processor, context_len, constants, conv_templates, mm_utils)


def make_prompt(bundle: LlavaBundle, question: str, conv_mode: str = "llava_v1") -> str:
    conv = copy.deepcopy(bundle.conv_templates[conv_mode])
    conv.append_message(conv.roles[0], bundle.constants.DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def make_supervised_prompt(bundle: LlavaBundle, question: str, answer: str, conv_mode: str = "llava_v1") -> str:
    conv = copy.deepcopy(bundle.conv_templates[conv_mode])
    conv.append_message(conv.roles[0], bundle.constants.DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], answer)
    return conv.get_prompt()


def tokenize_prompt(bundle: LlavaBundle, prompt: str) -> torch.Tensor:
    return bundle.mm_utils.tokenizer_image_token(
        prompt,
        bundle.tokenizer,
        bundle.constants.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )


def process_image(bundle: LlavaBundle, image_path: str | Path, device: torch.device, dtype=torch.float16) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_tensor = bundle.mm_utils.process_images([image], bundle.image_processor, bundle.model.config)[0]
    return image_tensor.unsqueeze(0).to(device=device, dtype=dtype)


def base_model(model):
    if hasattr(model, "get_model"):
        return model.get_model()
    return model.model if hasattr(model, "model") else model


def embed_tokens(model):
    base = base_model(model)
    if hasattr(base, "embed_tokens"):
        return base.embed_tokens
    if hasattr(base, "get_input_embeddings"):
        return base.get_input_embeddings()
    raise AttributeError("Could not find language-model token embeddings.")


@torch.no_grad()
def extract_visual_tokens(model, image_tensor: torch.Tensor) -> torch.Tensor:
    base = base_model(model)
    if hasattr(base, "get_vision_tower"):
        vision_tower = base.get_vision_tower()
    elif hasattr(base, "vision_tower"):
        vision_tower = base.vision_tower
    else:
        raise AttributeError("Could not find LLaVA vision tower.")

    image_features = vision_tower(image_tensor)
    if isinstance(image_features, (list, tuple)):
        image_features = image_features[0]
    if hasattr(base, "mm_projector"):
        projector = base.mm_projector
        param = next(projector.parameters(), None)
        if param is not None:
            image_features = image_features.to(dtype=param.dtype, device=param.device)
        image_features = projector(image_features)
    return image_features


def image_token_positions(input_ids: torch.Tensor, image_token_index: int) -> list[int | None]:
    positions = []
    for row in input_ids:
        loc = (row.detach().cpu() == image_token_index).nonzero(as_tuple=True)[0]
        positions.append(int(loc[0]) if len(loc) else None)
    return positions


def build_inputs_embeds_with_visual_tokens(
    model,
    input_ids: torch.Tensor,
    visual_tokens: torch.Tensor,
    positions: list[int | None],
):
    token_embed = embed_tokens(model)
    device = visual_tokens.device
    rows = []
    masks = []
    for row, visual, pos in zip(input_ids, visual_tokens, positions):
        row = row.to(device)
        if pos is None:
            embeds = token_embed(row)
        else:
            before = token_embed(row[:pos]) if pos > 0 else visual.new_empty((0, visual.shape[-1]))
            after = token_embed(row[pos + 1 :]) if pos + 1 < row.shape[0] else visual.new_empty((0, visual.shape[-1]))
            embeds = torch.cat([before, visual, after], dim=0)
        rows.append(embeds)
        masks.append(torch.ones(embeds.shape[0], dtype=torch.long, device=device))

    max_len = max(row.shape[0] for row in rows)
    padded_rows = []
    padded_masks = []
    for row, mask in zip(rows, masks):
        pad = max_len - row.shape[0]
        if pad:
            row = torch.cat([row, row.new_zeros((pad, row.shape[-1]))], dim=0)
            mask = torch.cat([mask, mask.new_zeros((pad,))], dim=0)
        padded_rows.append(row)
        padded_masks.append(mask)
    return torch.stack(padded_rows), torch.stack(padded_masks)


@torch.no_grad()
def generate_baseline(
    bundle: LlavaBundle,
    question: str,
    image_path: str | Path,
    device: torch.device,
    conv_mode: str = "llava_v1",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    prompt = make_prompt(bundle, question, conv_mode)
    input_ids = tokenize_prompt(bundle, prompt).unsqueeze(0).to(device)
    image_tensor = process_image(bundle, image_path, device)
    gen_kwargs = {
        "images": image_tensor,
        "do_sample": temperature > 0,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    outputs = bundle.model.generate(input_ids, **gen_kwargs)
    return bundle.tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_with_refined_tokens(
    bundle: LlavaBundle,
    question: str,
    image_path: str | Path,
    adapter,
    class_embeddings: torch.Tensor,
    device: torch.device,
    conv_mode: str = "llava_v1",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    prompt = make_prompt(bundle, question, conv_mode)
    input_ids = tokenize_prompt(bundle, prompt).unsqueeze(0).to(device)
    image_tensor = process_image(bundle, image_path, device)
    visual_tokens = extract_visual_tokens(bundle.model, image_tensor)
    refined_tokens, _, _ = adapter(visual_tokens.float(), class_embeddings.to(device=device, dtype=torch.float32))
    refined_tokens = refined_tokens.to(dtype=visual_tokens.dtype)
    positions = image_token_positions(input_ids, bundle.constants.IMAGE_TOKEN_INDEX)
    current_embeds, _ = build_inputs_embeds_with_visual_tokens(bundle.model, input_ids, refined_tokens, positions)

    token_embed = embed_tokens(bundle.model)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        outputs = bundle.model(inputs_embeds=current_embeds, return_dict=True, use_cache=False)
        logits = outputs.logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)
        if next_id.item() == bundle.tokenizer.eos_token_id:
            break
        generated.append(next_id.item())
        current_embeds = torch.cat([current_embeds, token_embed(next_id)], dim=1)
    return bundle.tokenizer.decode(generated, skip_special_tokens=True).strip()
