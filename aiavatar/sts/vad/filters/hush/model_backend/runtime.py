"""Python Hush runtime with shared inference and session-owned DSP/model state."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .dsp import DspState


@dataclass
class PythonState:
    dsp: DspState
    inputs: dict


class PythonModel:
    def __init__(self, model_path, atten_lim_db):
        import onnxruntime as ort

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Hush asset not found: {path}")
        if path.suffix == ".onnx":
            model = str(path)
        else:
            from .graph import build_streaming_model
            model = build_streaming_model(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("hush.format_version") != "1":
            raise ValueError("Use a Hush streaming ONNX produced by hush.model_backend.graph, or the official bundle")
        config = json.loads(metadata["hush.config"])
        expected = {"sr": 16000, "hop_size": 160, "fft_size": 320, "nb_erb": 32,
                    "nb_df": 64, "df_order": 5, "min_nb_erb_freqs": 2,
                    "norm_tau": 1.0, "conv_lookahead": 0, "df_lookahead": 0}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Unsupported Hush streaming model geometry")
        self.states = json.loads(metadata["hush.states"])
        inputs = {item.name: item for item in self.session.get_inputs()}
        outputs = self.session.get_outputs()
        expected_inputs = {"feat_erb": [1, 1, 1, 32], "feat_spec": [1, 2, 1, 64]}
        if len(self.states) != 7 or len({s["input"] for s in self.states}) != 7:
            raise ValueError("Unsupported Hush streaming model states")
        for state in self.states:
            if (state["dtype"] != "float32" or state["stage"] not in ("enc", "erb", "df")
                    or not state["shape"] or any(type(n) is not int or n < 1 for n in state["shape"])):
                raise ValueError("Invalid Hush streaming state metadata")
            expected_inputs[state["input"]] = state["shape"]
        if set(inputs) != set(expected_inputs) or any(
            inputs[name].type != "tensor(float)" or inputs[name].shape != shape
            for name, shape in expected_inputs.items()
        ):
            raise ValueError("Invalid Hush streaming model inputs")
        expected_outputs = [("mask", [1, 1, 1, 32]), ("coefs", [1, 1, 64, 10]), ("lsnr", [1, 1, 1])]
        expected_outputs += [(state["output"], state["shape"]) for state in self.states]
        if len(outputs) != len(expected_outputs) or any(
            node.name != name or node.type != "tensor(float)" or node.shape != shape
            for node, (name, shape) in zip(outputs, expected_outputs)
        ):
            raise ValueError("Invalid Hush streaming model outputs")
        self.attenuation_limit = 10 ** (-atten_lim_db / 20) if atten_lim_db < 100 else 0.0

    def create_state(self):
        return PythonState(DspState(self.attenuation_limit), {
            state["input"]: np.zeros(state["shape"], dtype=np.float32) for state in self.states
        })

    def process_frame(self, state, audio):
        state.inputs.update(state.dsp.analyze(audio))
        mask, coefs, lsnr, *next_states = self.session.run(None, state.inputs)
        snr = float(lsnr.item())
        if not np.isfinite(snr):
            raise RuntimeError("Hush produced a non-finite SNR")
        # Match the native ABI's -15 / 35 dB stage thresholds. Its decoder
        # histories pause when a stage is skipped; the encoder always advances.
        apply_decoders = -15.0 <= snr <= 35.0
        for info, value in zip(self.states, next_states):
            if info["stage"] == "enc" or apply_decoders:
                state.inputs[info["input"]] = value
        if snr < -15.0:
            mask, coefs = np.zeros_like(mask), None
        elif snr > 35.0:
            mask, coefs = None, None
        return state.dsp.synthesize(mask, coefs)
