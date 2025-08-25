"""
Script to convert T2IAdapters, COAdapters and Fusers into pipeline usable checkpoints
"""
import sys
import torch
import argparse

from pathlib import Path
from diffusers import T2IAdapter

sys.path.append(str(Path(__file__).parents[4]))
from stg_research.diffusion_models.condition_diffusion.models.adapter_fusion import Fuser, T2IAdapterXL

# -- Conversion Functions -- #
def convert_full_adapter(src_state, in_channels):
    # Get body length
    original_body_length = max([int(x.split(".")[1]) for x in src_state.keys() if "body." in x]) + 1
    # Check state is not empty
    assert len(list(src_state.keys())) != 0, "Make sure that the model checkpoint is not corrupt."
    # Check checkpoint has the same shape as the Adapter architecture
    assert original_body_length == 8
    assert src_state["body.0.block1.weight"].shape == (320, 320, 3, 3) # (0, 1) -> channels 1
    assert src_state["body.2.in_conv.weight"].shape == (640, 320, 1, 1) # (2, 3) -> channels 2
    assert src_state["body.4.in_conv.weight"].shape == (1280, 640, 1, 1) # (4, 5) -> channels 3
    assert src_state["body.6.block1.weight"].shape == (1280, 1280, 3, 3) # (6, 7) -> channels 4
    # Mapping
    dst_state = {
        "adapter.conv_in.weight": src_state.pop("conv_in.weight"),
        "adapter.conv_in.bias": src_state.pop("conv_in.bias"),
        # 0.resnets.0
        "adapter.body.0.resnets.0.block1.weight": src_state.pop("body.0.block1.weight"),
        "adapter.body.0.resnets.0.block1.bias": src_state.pop("body.0.block1.bias"),
        "adapter.body.0.resnets.0.block2.weight": src_state.pop("body.0.block2.weight"),
        "adapter.body.0.resnets.0.block2.bias": src_state.pop("body.0.block2.bias"),
        # 0.resnets.1
        "adapter.body.0.resnets.1.block1.weight": src_state.pop("body.1.block1.weight"),
        "adapter.body.0.resnets.1.block1.bias": src_state.pop("body.1.block1.bias"),
        "adapter.body.0.resnets.1.block2.weight": src_state.pop("body.1.block2.weight"),
        "adapter.body.0.resnets.1.block2.bias": src_state.pop("body.1.block2.bias"),
        # 1
        "adapter.body.1.in_conv.weight": src_state.pop("body.2.in_conv.weight"),
        "adapter.body.1.in_conv.bias": src_state.pop("body.2.in_conv.bias"),
        # 1.resnets.0
        "adapter.body.1.resnets.0.block1.weight": src_state.pop("body.2.block1.weight"),
        "adapter.body.1.resnets.0.block1.bias": src_state.pop("body.2.block1.bias"),
        "adapter.body.1.resnets.0.block2.weight": src_state.pop("body.2.block2.weight"),
        "adapter.body.1.resnets.0.block2.bias": src_state.pop("body.2.block2.bias"),
        # 1.resnets.1
        "adapter.body.1.resnets.1.block1.weight": src_state.pop("body.3.block1.weight"),
        "adapter.body.1.resnets.1.block1.bias": src_state.pop("body.3.block1.bias"),
        "adapter.body.1.resnets.1.block2.weight": src_state.pop("body.3.block2.weight"),
        "adapter.body.1.resnets.1.block2.bias": src_state.pop("body.3.block2.bias"),
        # 2
        "adapter.body.2.in_conv.weight": src_state.pop("body.4.in_conv.weight"),
        "adapter.body.2.in_conv.bias": src_state.pop("body.4.in_conv.bias"),
        # 2.resnets.0
        "adapter.body.2.resnets.0.block1.weight": src_state.pop("body.4.block1.weight"),
        "adapter.body.2.resnets.0.block1.bias": src_state.pop("body.4.block1.bias"),
        "adapter.body.2.resnets.0.block2.weight": src_state.pop("body.4.block2.weight"),
        "adapter.body.2.resnets.0.block2.bias": src_state.pop("body.4.block2.bias"),
        # 2.resnets.1
        "adapter.body.2.resnets.1.block1.weight": src_state.pop("body.5.block1.weight"),
        "adapter.body.2.resnets.1.block1.bias": src_state.pop("body.5.block1.bias"),
        "adapter.body.2.resnets.1.block2.weight": src_state.pop("body.5.block2.weight"),
        "adapter.body.2.resnets.1.block2.bias": src_state.pop("body.5.block2.bias"),
        # 3.resnets.0
        "adapter.body.3.resnets.0.block1.weight": src_state.pop("body.6.block1.weight"),
        "adapter.body.3.resnets.0.block1.bias": src_state.pop("body.6.block1.bias"),
        "adapter.body.3.resnets.0.block2.weight": src_state.pop("body.6.block2.weight"),
        "adapter.body.3.resnets.0.block2.bias": src_state.pop("body.6.block2.bias"),
        # 3.resnets.1
        "adapter.body.3.resnets.1.block1.weight": src_state.pop("body.7.block1.weight"),
        "adapter.body.3.resnets.1.block1.bias": src_state.pop("body.7.block1.bias"),
        "adapter.body.3.resnets.1.block2.weight": src_state.pop("body.7.block2.weight"),
        "adapter.body.3.resnets.1.block2.bias": src_state.pop("body.7.block2.bias"),
    }
    # Loading Adapting
    adapter = T2IAdapter(in_channels=in_channels, adapter_type="full_adapter")
    adapter.load_state_dict(dst_state)
    return adapter

def convert_light_adapter(src_state):
    # Get body length
    original_body_length = max([int(x.split(".")[1]) for x in src_state.keys() if "body." in x]) + 1
     # Check state is not empty
    assert len(list(src_state.keys())) != 0, "Make sure that the model checkpoint is not corrupt."
    # Check checkpoint has the same shape as the Adapter architecture
    assert original_body_length == 4
    # Mapping
    dst_state = {
        # body.0.in_conv
        "adapter.body.0.in_conv.weight": src_state.pop("body.0.in_conv.weight"),
        "adapter.body.0.in_conv.bias": src_state.pop("body.0.in_conv.bias"),
        # body.0.resnets.0
        "adapter.body.0.resnets.0.block1.weight": src_state.pop("body.0.body.0.block1.weight"),
        "adapter.body.0.resnets.0.block1.bias": src_state.pop("body.0.body.0.block1.bias"),
        "adapter.body.0.resnets.0.block2.weight": src_state.pop("body.0.body.0.block2.weight"),
        "adapter.body.0.resnets.0.block2.bias": src_state.pop("body.0.body.0.block2.bias"),
        # body.0.resnets.1
        "adapter.body.0.resnets.1.block1.weight": src_state.pop("body.0.body.1.block1.weight"),
        "adapter.body.0.resnets.1.block1.bias": src_state.pop("body.0.body.1.block1.bias"),
        "adapter.body.0.resnets.1.block2.weight": src_state.pop("body.0.body.1.block2.weight"),
        "adapter.body.0.resnets.1.block2.bias": src_state.pop("body.0.body.1.block2.bias"),
        # body.0.resnets.2
        "adapter.body.0.resnets.2.block1.weight": src_state.pop("body.0.body.2.block1.weight"),
        "adapter.body.0.resnets.2.block1.bias": src_state.pop("body.0.body.2.block1.bias"),
        "adapter.body.0.resnets.2.block2.weight": src_state.pop("body.0.body.2.block2.weight"),
        "adapter.body.0.resnets.2.block2.bias": src_state.pop("body.0.body.2.block2.bias"),
        # body.0.resnets.3
        "adapter.body.0.resnets.3.block1.weight": src_state.pop("body.0.body.3.block1.weight"),
        "adapter.body.0.resnets.3.block1.bias": src_state.pop("body.0.body.3.block1.bias"),
        "adapter.body.0.resnets.3.block2.weight": src_state.pop("body.0.body.3.block2.weight"),
        "adapter.body.0.resnets.3.block2.bias": src_state.pop("body.0.body.3.block2.bias"),
        # body.0.out_conv
        "adapter.body.0.out_conv.weight": src_state.pop("body.0.out_conv.weight"),
        "adapter.body.0.out_conv.bias": src_state.pop("body.0.out_conv.bias"),
        # body.1.in_conv
        "adapter.body.1.in_conv.weight": src_state.pop("body.1.in_conv.weight"),
        "adapter.body.1.in_conv.bias": src_state.pop("body.1.in_conv.bias"),
        # body.1.resnets.0
        "adapter.body.1.resnets.0.block1.weight": src_state.pop("body.1.body.0.block1.weight"),
        "adapter.body.1.resnets.0.block1.bias": src_state.pop("body.1.body.0.block1.bias"),
        "adapter.body.1.resnets.0.block2.weight": src_state.pop("body.1.body.0.block2.weight"),
        "adapter.body.1.resnets.0.block2.bias": src_state.pop("body.1.body.0.block2.bias"),
        # body.1.resnets.1
        "adapter.body.1.resnets.1.block1.weight": src_state.pop("body.1.body.1.block1.weight"),
        "adapter.body.1.resnets.1.block1.bias": src_state.pop("body.1.body.1.block1.bias"),
        "adapter.body.1.resnets.1.block2.weight": src_state.pop("body.1.body.1.block2.weight"),
        "adapter.body.1.resnets.1.block2.bias": src_state.pop("body.1.body.1.block2.bias"),
        # body.1.body.2
        "adapter.body.1.resnets.2.block1.weight": src_state.pop("body.1.body.2.block1.weight"),
        "adapter.body.1.resnets.2.block1.bias": src_state.pop("body.1.body.2.block1.bias"),
        "adapter.body.1.resnets.2.block2.weight": src_state.pop("body.1.body.2.block2.weight"),
        "adapter.body.1.resnets.2.block2.bias": src_state.pop("body.1.body.2.block2.bias"),
        # body.1.body.3
        "adapter.body.1.resnets.3.block1.weight": src_state.pop("body.1.body.3.block1.weight"),
        "adapter.body.1.resnets.3.block1.bias": src_state.pop("body.1.body.3.block1.bias"),
        "adapter.body.1.resnets.3.block2.weight": src_state.pop("body.1.body.3.block2.weight"),
        "adapter.body.1.resnets.3.block2.bias": src_state.pop("body.1.body.3.block2.bias"),
        # body.1.out_conv
        "adapter.body.1.out_conv.weight": src_state.pop("body.1.out_conv.weight"),
        "adapter.body.1.out_conv.bias": src_state.pop("body.1.out_conv.bias"),
        # body.2.in_conv
        "adapter.body.2.in_conv.weight": src_state.pop("body.2.in_conv.weight"),
        "adapter.body.2.in_conv.bias": src_state.pop("body.2.in_conv.bias"),
        # body.2.body.0
        "adapter.body.2.resnets.0.block1.weight": src_state.pop("body.2.body.0.block1.weight"),
        "adapter.body.2.resnets.0.block1.bias": src_state.pop("body.2.body.0.block1.bias"),
        "adapter.body.2.resnets.0.block2.weight": src_state.pop("body.2.body.0.block2.weight"),
        "adapter.body.2.resnets.0.block2.bias": src_state.pop("body.2.body.0.block2.bias"),
        # body.2.body.1
        "adapter.body.2.resnets.1.block1.weight": src_state.pop("body.2.body.1.block1.weight"),
        "adapter.body.2.resnets.1.block1.bias": src_state.pop("body.2.body.1.block1.bias"),
        "adapter.body.2.resnets.1.block2.weight": src_state.pop("body.2.body.1.block2.weight"),
        "adapter.body.2.resnets.1.block2.bias": src_state.pop("body.2.body.1.block2.bias"),
        # body.2.body.2
        "adapter.body.2.resnets.2.block1.weight": src_state.pop("body.2.body.2.block1.weight"),
        "adapter.body.2.resnets.2.block1.bias": src_state.pop("body.2.body.2.block1.bias"),
        "adapter.body.2.resnets.2.block2.weight": src_state.pop("body.2.body.2.block2.weight"),
        "adapter.body.2.resnets.2.block2.bias": src_state.pop("body.2.body.2.block2.bias"),
        # body.2.body.3
        "adapter.body.2.resnets.3.block1.weight": src_state.pop("body.2.body.3.block1.weight"),
        "adapter.body.2.resnets.3.block1.bias": src_state.pop("body.2.body.3.block1.bias"),
        "adapter.body.2.resnets.3.block2.weight": src_state.pop("body.2.body.3.block2.weight"),
        "adapter.body.2.resnets.3.block2.bias": src_state.pop("body.2.body.3.block2.bias"),
        # body.2.out_conv
        "adapter.body.2.out_conv.weight": src_state.pop("body.2.out_conv.weight"),
        "adapter.body.2.out_conv.bias": src_state.pop("body.2.out_conv.bias"),
        # body.3.in_conv
        "adapter.body.3.in_conv.weight": src_state.pop("body.3.in_conv.weight"),
        "adapter.body.3.in_conv.bias": src_state.pop("body.3.in_conv.bias"),
        # body.3.body.0
        "adapter.body.3.resnets.0.block1.weight": src_state.pop("body.3.body.0.block1.weight"),
        "adapter.body.3.resnets.0.block1.bias": src_state.pop("body.3.body.0.block1.bias"),
        "adapter.body.3.resnets.0.block2.weight": src_state.pop("body.3.body.0.block2.weight"),
        "adapter.body.3.resnets.0.block2.bias": src_state.pop("body.3.body.0.block2.bias"),
        # body.3.body.1
        "adapter.body.3.resnets.1.block1.weight": src_state.pop("body.3.body.1.block1.weight"),
        "adapter.body.3.resnets.1.block1.bias": src_state.pop("body.3.body.1.block1.bias"),
        "adapter.body.3.resnets.1.block2.weight": src_state.pop("body.3.body.1.block2.weight"),
        "adapter.body.3.resnets.1.block2.bias": src_state.pop("body.3.body.1.block2.bias"),
        # body.3.body.2
        "adapter.body.3.resnets.2.block1.weight": src_state.pop("body.3.body.2.block1.weight"),
        "adapter.body.3.resnets.2.block1.bias": src_state.pop("body.3.body.2.block1.bias"),
        "adapter.body.3.resnets.2.block2.weight": src_state.pop("body.3.body.2.block2.weight"),
        "adapter.body.3.resnets.2.block2.bias": src_state.pop("body.3.body.2.block2.bias"),
        # body.3.body.3
        "adapter.body.3.resnets.3.block1.weight": src_state.pop("body.3.body.3.block1.weight"),
        "adapter.body.3.resnets.3.block1.bias": src_state.pop("body.3.body.3.block1.bias"),
        "adapter.body.3.resnets.3.block2.weight": src_state.pop("body.3.body.3.block2.weight"),
        "adapter.body.3.resnets.3.block2.bias": src_state.pop("body.3.body.3.block2.bias"),
        # body.3.out_conv
        "adapter.body.3.out_conv.weight": src_state.pop("body.3.out_conv.weight"),
        "adapter.body.3.out_conv.bias": src_state.pop("body.3.out_conv.bias"),
    }
    # Loading Adapter
    adapter = T2IAdapter(in_channels=3, channels=[320, 640, 1280], num_res_blocks=4, adapter_type="light_adapter")
    adapter.load_state_dict(dst_state)
    return adapter

def convert_style_adapter(src_state):
    #TODO: implement the conversion
    raise NotImplementedError

def convert_fuser(src_state):
     # Mapping
    dst_state = {
        name.replace(
            'transformer_layes', 'transformer_layers'
        ) if 'transformer_layes' in name else name: module for name, module in src_state.items()
    }

    # Loading Fuser
    fuser = Fuser()
    fuser.load_state_dict(dst_state)
    return fuser

def convert_adapter_xl(src_state):
    # Loading Adapter
    adapter = T2IAdapterXL(in_channels=1, channels=[320, 640, 1280, 1280], num_res_blocks=2, downscale_factor=16)
    adapter.load_state_dict(src_state)
    return adapter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--output_path", default=None, type=str, required=True, help="Path to the store the result checkpoint."
    )
    parser.add_argument(
        "--adapter_type", choices=["full", "light", "style", "fuser", "xl"], required=True,
        help="The type of adapter. Can be Full (Canny), Light (Color), Style or the Fuser",
    )
    parser.add_argument("--in_channels", required=False, type=int, help="Input channels for full adapter")
    arguments = parser.parse_args()
    
    # Load state dictionary 
    src_state = torch.load(arguments.checkpoint_path)

    if arguments.adapter_type == "full":
        assert arguments.in_channels is not None, "set `--in_channels=<n>`"
        model = convert_full_adapter(src_state, arguments.in_channels)
    if arguments.adapter_type == "light":
        model = convert_light_adapter(src_state)
    if arguments.adapter_type == "style":
        model = convert_style_adapter(src_state)
    if arguments.adapter_type == "fuser":
        model = convert_fuser(src_state)
    if arguments.adapter_type == 'xl':
        model = convert_adapter_xl(src_state)

    model.save_pretrained(arguments.output_path)
    print('Conversion performed successfully!')