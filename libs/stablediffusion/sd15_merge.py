#!/usr/bin/env python

# load libs
try:
    import torch
    from libs.shared.utils import get_gpu
    from pathlib import PosixPath
    from safetensors.torch import load_file, save_file
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

# load checkpoint weight and return a dict
def load_checkpoint_dict(checkpoint_file: PosixPath, device: str = "cpu") -> dict:
    assert checkpoint_file.name.endswith('.safetensors')

    # return network weights dict
    print(f"Loading weights from checkpoint {checkpoint_file}...")
    return load_file(checkpoint_file, device=device)

# compare checkpoint keys
def compare_checkpoint_keys(base_ckpt: dict, compare_ckpt: dict) -> bool:
    # compare layers
    for key in base_ckpt.keys():
        if key not in compare_ckpt:
            return False
    return True

# merge checkpoints and return resulting dict
def merge_checkpoints_linear(base_ckpt: dict, additional_ckpt: dict, alpha: float = 0.5) -> dict:
    merged_dict = {}

    # linear interpolation
    for key in base_ckpt.keys():
        layer_a = base_ckpt[key].to(torch.float32)
        layer_b = additional_ckpt[key].to(torch.float32)

        # type
        if type(alpha) == str:
            alpha = float(alpha)

        # merge, linear: alpha * A + (1 - alpha) * B
        merged_dict[key] = ((layer_a * alpha) + ((1-alpha) * layer_b)).to(layer_a.dtype)

    return merged_dict

# save resulting checkpoint
def save_checkpoint(final_ckpt: dict, filename: str = "merged_output.safetensors") -> None:
    # save resulting weights to disk
    print(f"Saving checkpoint to {filename}...")
    f = PosixPath(filename)
    try:
        save_file(final_ckpt, f)
    except Exception:
        return False

    # ok
    return True
