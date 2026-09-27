"""Convert the supported Hush 16 kHz export into a stateful, one-hop graph.

The upstream export has five GRUs and two causal temporal convolutions. Their
history becomes explicit graph inputs/outputs so one ONNX Runtime session can
share weights between independent audio streams. Conversion requires ``onnx``;
loading the resulting bytes in ONNX Runtime does not.

This deliberately supports the architecture of
``advanced_dfnet16k_model_best_onnx.tar.gz``, not arbitrary DeepFilterNet exports.
Its structure is pinned independently of the trained initializer values: a
different architecture/export fails instead of silently losing temporal state.
"""

import configparser
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile


FORMAT_VERSION = "1"
STATE_METADATA_KEY = "hush.states"
CONFIG_METADATA_KEY = "hush.config"
VERSION_METADATA_KEY = "hush.format_version"

_MEMBERS = ("enc.onnx", "erb_dec.onnx", "df_dec.onnx")
# SHA-256 of GraphProto with each weight initializer replaced by name/type/shape.
# Constants inside nodes remain included (padding, reshape axes, zero states).
# Reference: weya-ai/hush model revision 40812c28145510d8a4b14641bb58c879a7a7b4fe.
_STRUCTURES = (
    "58c39f7e98a1196a3a73a20e0dbfaef61b1d5be221c4b752f3e59a556a3be12c",
    "7409667de7dc4ad0bbb8f4e17d3bf988bb0d6217b03436c9b9adc1d034979605",
    "556647b8c940ca8c7ff646c807419ed5820624e687861e67471df8a85c15d34d",
)
_DF_CONFIG = {
    "sr": 16000, "hop_size": 160, "fft_size": 320, "nb_erb": 32,
    "nb_df": 64, "min_nb_erb_freqs": 2, "norm_tau": 1.0,
}
_NET_CONFIG = {"conv_ch": 16, "conv_lookahead": 0, "df_lookahead": 0, "df_order": 5}


def _structure_digest(model):
    graph = deepcopy(model.graph)
    for tensor in graph.initializer:
        tensor.CopyFrom(type(tensor)(name=tensor.name, data_type=tensor.data_type, dims=tensor.dims))
    return hashlib.sha256(graph.SerializeToString(deterministic=True)).hexdigest()


def _read_bundle(bundle_path):
    with tarfile.open(bundle_path, "r:gz") as archive:
        # Named member reads only: never extract archive paths to the filesystem.
        config_bytes = archive.extractfile("config.ini").read()
        contents = [archive.extractfile(name).read() for name in _MEMBERS]
    config = configparser.ConfigParser()
    config.read_string(config_bytes.decode("utf-8"))
    for section, expected in (("df", _DF_CONFIG), ("deepfilternet", _NET_CONFIG)):
        for key, value in expected.items():
            if not config.has_option(section, key) or config.getfloat(section, key) != value:
                raise ValueError(f"Unsupported Hush model: [{section}] {key} must be {value}")
    if config.get("train", "model", fallback="") != "deepfilternet3":
        raise ValueError("Unsupported Hush model: expected DeepFilterNet3")
    return contents


def _prune(graph):
    """Remove now-unused padding/initial-hidden subgraphs and their constants."""
    needed = {value.name for value in graph.output}
    kept = []
    for node in reversed(graph.node):
        if needed.intersection(node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    initializers = [tensor for tensor in graph.initializer if tensor.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(initializers)


def build_streaming_model(bundle_path: str | Path) -> bytes:
    """Return a validated one-frame ONNX graph with explicit, zero-initialized state.

    Audio feature inputs are ``feat_erb`` [1,1,1,32] and ``feat_spec`` [1,2,1,64].
    The first outputs are ``mask`` [1,1,1,32], ``coefs`` [1,1,64,10], and ``lsnr``
    [1,1,1]. Remaining ``state_*`` inputs correspond to ``next_state_*`` outputs.
    ``hush.states`` model metadata is a JSON list of input/output/shape/dtype/stage
    specifications; each stream starts with zeros and retains its own outputs.

    All three network stages execute. Callers implementing native LSNR-based
    decoder skipping must retain the old state for any skipped decoder stage.
    Conversion performs filesystem/CPU work and belongs off the asyncio loop.
    """
    try:
        import onnx
        from onnx import checker, compose, helper
    except ImportError as exc:
        raise ImportError(
            "Converting a Hush bundle requires 'onnx': install aiavatar[hush-python], "
            "or use a preconverted streaming .onnx file"
        ) from exc

    contents = _read_bundle(bundle_path)
    models = [onnx.load_model_from_string(content) for content in contents]
    for name, model, expected in zip(_MEMBERS, models, _STRUCTURES):
        if model.ir_version != 7 or [(op.domain, op.version) for op in model.opset_import] != [("", 14)]:
            raise ValueError(f"Unsupported Hush model {name}: expected IR 7 / opset 14")
        if model.functions or model.training_info or model.graph.sparse_initializer:
            raise ValueError(f"Unsupported Hush model {name}: unexpected model extensions")
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
            raise ValueError(f"Unsupported Hush model {name}: external tensor data")
        if _structure_digest(model) != expected:
            raise ValueError(f"Unsupported Hush graph architecture/export: {name}")
        checker.check_model(model, full_check=True)

    stages = ("enc", "erb", "df")
    prefixed = [compose.add_prefix(model, stage + "/") for model, stage in zip(models, stages)]
    nodes, initializers, states = [], [], []
    inputs = [helper.make_tensor_value_info("feat_erb", onnx.TensorProto.FLOAT, [1, 1, 1, 32]),
              helper.make_tensor_value_info("feat_spec", onnx.TensorProto.FLOAT, [1, 2, 1, 64])]
    state_outputs = []

    def state_spec(label, shape, stage):
        name, next_name = "state_" + label, "next_state_" + label
        inputs.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
        state_outputs.append(helper.make_tensor_value_info(next_name, onnx.TensorProto.FLOAT, shape))
        states.append({"input": name, "output": next_name, "shape": shape,
                       "dtype": "float32", "stage": stage})
        return name, next_name

    for model, stage in zip(prefixed, stages):
        initializers.extend(model.graph.initializer)
        for original in model.graph.input:
            name = original.name.split("/", 1)[1]
            source = name if stage == "enc" else "enc/" + name
            nodes.append(helper.make_node("Identity", [source], [original.name]))
        gru_index = 0
        for node in model.graph.node:
            if node.op_type == "Pad":
                # Both pinned Pad nodes prepend two zero time frames to raw
                # encoder features. Concat prior features instead, then keep
                # the newest two frames for the next invocation.
                feature = node.input[0].split("/", 1)[1]
                channels, bins = (1, 32) if feature == "feat_erb" else (2, 64)
                label = "enc_conv_" + feature.removeprefix("feat_")
                name, next_name = state_spec(label, [1, channels, 2, bins], "enc")
                nodes.append(helper.make_node("Concat", [name, node.input[0]], list(node.output), axis=2))
                constants = []
                for part, value in (("start", 1), ("end", 3), ("axis", 2)):
                    constant = name + "_" + part
                    initializers.append(helper.make_tensor(constant, onnx.TensorProto.INT64, [1], [value]))
                    constants.append(constant)
                nodes.append(helper.make_node("Slice", [node.output[0], *constants], [next_name]))
            elif node.op_type == "GRU":
                name, next_name = state_spec(f"{stage}_gru_{gru_index}", [1, 1, 256], stage)
                gru_index += 1
                node.input[5] = name
                nodes.append(node)
                nodes.append(helper.make_node("Identity", [node.output[1]], [next_name]))
            else:
                nodes.append(node)

    outputs = []
    for source, name, shape in (("erb/m", "mask", [1, 1, 1, 32]),
                                ("df/coefs", "coefs", [1, 1, 64, 10]),
                                ("enc/lsnr", "lsnr", [1, 1, 1])):
        nodes.append(helper.make_node("Identity", [source], [name]))
        outputs.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    graph = helper.make_graph(nodes, "hush_streaming_16k", inputs, outputs + state_outputs,
                              initializer=initializers)
    _prune(graph)
    model = helper.make_model(graph, producer_name="AIAvatarKit Hush streaming conversion",
                              ir_version=7, opset_imports=[helper.make_opsetid("", 14)])
    helper.set_model_props(model, {
        VERSION_METADATA_KEY: FORMAT_VERSION,
        CONFIG_METADATA_KEY: json.dumps({**_DF_CONFIG, **_NET_CONFIG}, sort_keys=True),
        STATE_METADATA_KEY: json.dumps(states),
    })
    checker.check_model(model, full_check=True)
    return model.SerializeToString()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Supported Hush model tar.gz")
    parser.add_argument("--output", type=Path, required=True, help="New streaming .onnx file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    data = build_streaming_model(args.input)
    with args.output.open("xb") as output:
        output.write(data)
