"""Drop-in TRT-backed replacement for diffusers UNet2DConditionModel.

Same input/output API as ORTUNetWrapper (evals/onnx_eval.py:33) so the
diffusers SDXL pipeline doesn't know it's wrapping a TRT engine.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import tensorrt as trt
import torch
import torch.nn as nn


_INPUT_SHAPES = {
    "sample": (1, 4, 128, 128),
    "timestep": (1,),
    "encoder_hidden_states": (1, 77, 2048),
    "text_embeds": (1, 1280),
    "time_ids": (1, 6),
}
_OUTPUT_NAME = "noise_pred"
_OUTPUT_SHAPE = (1, 4, 128, 128)

_TRT_TO_TORCH = {
    trt.DataType.FLOAT:  torch.float32,
    trt.DataType.HALF:   torch.float16,
    trt.DataType.INT64:  torch.int64,
    trt.DataType.INT32:  torch.int32,
    trt.DataType.INT8:   torch.int8,
    trt.DataType.BOOL:   torch.bool,
}


class TRTUNetWrapper(nn.Module):
    def __init__(self, engine_path: Path, reference_unet_config_dict: dict,
                 device: str = "cuda"):
        super().__init__()
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        runtime = trt.Runtime(logger)
        eng_bytes = Path(engine_path).read_bytes()
        self.engine = runtime.deserialize_cuda_engine(eng_bytes)
        self.context = self.engine.create_execution_context()
        self._device = torch.device(device)
        self._dtype = torch.float16

        # Pre-allocate I/O buffers on device. Engine was built with static
        # shapes so we know everything up front.
        self._in_buffers = {}
        for name, shape in _INPUT_SHAPES.items():
            trt_dt = self.engine.get_tensor_dtype(name)
            torch_dt = _TRT_TO_TORCH[trt_dt]
            buf = torch.empty(shape, dtype=torch_dt, device=self._device)
            self._in_buffers[name] = buf
            self.context.set_tensor_address(name, int(buf.data_ptr()))
        # Discover output tensor name dynamically — QDQ engines may rename it
        n_io = self.engine.num_io_tensors
        self._output_name = None
        for i in range(n_io):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self._output_name = name
                break
        if self._output_name is None:
            raise RuntimeError("TRT engine has no output tensor")
        print(f"[trt] output tensor: {self._output_name!r}")

        trt_out_dt = self.engine.get_tensor_dtype(self._output_name)
        self._out_buf = torch.empty(
            _OUTPUT_SHAPE, dtype=_TRT_TO_TORCH[trt_out_dt], device=self._device
        )
        self.context.set_tensor_address(self._output_name, int(self._out_buf.data_ptr()))
        self._stream = torch.cuda.Stream()

        # Stubs so diffusers SDXL pipeline checks pass
        from diffusers.configuration_utils import FrozenDict
        self.config = FrozenDict(reference_unet_config_dict)
        addition_time_embed_dim = reference_unet_config_dict.get(
            "addition_time_embed_dim", 256
        )
        in_features_expected = addition_time_embed_dim * 6 + 1280
        self.add_embedding = SimpleNamespace(
            linear_1=SimpleNamespace(in_features=in_features_expected)
        )

    @property
    def device(self): return self._device

    @property
    def dtype(self): return self._dtype

    def _copy_input(self, name: str, t: torch.Tensor):
        target = self._in_buffers[name]
        if name == "timestep":
            if not torch.is_tensor(t):
                t = torch.tensor([int(t)], dtype=torch.int64)
            if t.ndim == 0:
                t = t[None]
            target.copy_(t.to(device=self._device, dtype=torch.int64))
        else:
            target.copy_(t.to(device=self._device, dtype=target.dtype))

    def forward(self, sample, timestep, encoder_hidden_states,
                added_cond_kwargs=None, return_dict=True, **kwargs):
        if added_cond_kwargs is None:
            added_cond_kwargs = {}
        text_embeds = added_cond_kwargs["text_embeds"]
        time_ids = added_cond_kwargs["time_ids"]

        self._copy_input("sample", sample)
        self._copy_input("timestep", timestep)
        self._copy_input("encoder_hidden_states", encoder_hidden_states)
        self._copy_input("text_embeds", text_embeds)
        self._copy_input("time_ids", time_ids)

        self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        out = self._out_buf.to(self._dtype)
        if return_dict:
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
            return UNet2DConditionOutput(sample=out)
        return (out,)
