"""Load a compatible timm DeiT-Tiny backbone while retaining the fresh age head."""

import torch


def load_pretrained_weights(model, path):
    """Replace the model's backbone in place while retaining its fresh head.

    Read a tensor state dictionary on CPU. Require the complete backbone with
    matching shapes/dtypes; an optional source classifier is ignored.
    """

    source = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(source, dict):
        raise ValueError("Expected a timm DeiT-Tiny tensor state dictionary")
    mapped = {}
    expected = {"head.weight", "head.bias"}
    for name, target in model.state_dict().items():
        source_name = (name.replace("patch_embed.", "patch_embed.proj.")
                       .replace(".fc1.", ".mlp.fc1.").replace(".fc2.", ".mlp.fc2."))
        expected.add(source_name)
        if name.startswith("head."):
            mapped[name] = target
        else:
            if source_name not in source:
                raise ValueError(f"Missing pretrained tensor: {source_name}")
            value = source[source_name]
            if (not isinstance(value, torch.Tensor)
                    or value.shape != target.shape or value.dtype != target.dtype):
                raise ValueError(f"Incompatible pretrained tensor: {source_name}; "
                                 f"expected shape {tuple(target.shape)} and dtype {target.dtype}")
            mapped[name] = value
    if set(source) - expected:
        raise ValueError(f"Unexpected pretrained tensors: {sorted(set(source) - expected)}")
    model.load_state_dict(mapped, strict=True)
