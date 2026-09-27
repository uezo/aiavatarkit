"""Resolve official Hush assets; call from the model-loading worker."""

import hashlib
from pathlib import Path
import platform
import struct
import tempfile

import httpx


MODEL_REVISION = "40812c28145510d8a4b14641bb58c879a7a7b4fe"
MODEL_SHA256 = "45632ccaa82b71bb743d6caa7c78e983fe2f2790a3af7f6ec48e6ed7ba085df6"
MODEL_URL = (
    f"https://huggingface.co/weya-ai/hush/resolve/{MODEL_REVISION}/"
    "onnx/advanced_dfnet16k_model_best_onnx.tar.gz"
)
NATIVE_REVISION = "9f6414e91461a8f4bdf9840c0cdcdcb7da986339"
NATIVE_BASE_URL = f"https://raw.githubusercontent.com/pulp-vision/Hush/{NATIVE_REVISION}/deployment/lib/"
NATIVE_LIBRARIES = {
    ("Darwin", "arm64"): (
        "libweya_nc.dylib", "4bcd38634000d456ad68db4b6ff97fe4a462542aa65ac5f910fbf374ca33fbaa",
    ),
    ("Linux", "x86_64"): (
        "libweya_nc.so", "c4a5915c92e9500a49e6cd14804e6e88319277de2ef7d4d5cf6c0ac4456f3bbf",
    ),
    ("Windows", "x86_64"): (
        "weya_nc.dll", "633958ef839c94af6f8f5f9f8fb33c4daedaeec0373cb39a03934fdd7efa6a32",
    ),
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_path(model_path=None, cache_dir=None) -> Path:
    """Return an explicit local model, or download and verify the pinned bundle.

    This is synchronous and performs filesystem/network I/O. Explicit paths
    never download or use the cache. The default cache is ~/.cache/aiavatar/hush;
    only the original archive is saved, not the converted streaming ONNX model.
    Concurrent downloads may duplicate work but publish only verified files.
    """
    if model_path is not None:
        return _local_path(model_path)
    return _download_asset(MODEL_URL, MODEL_SHA256, f"hush-16k-{MODEL_SHA256}.tar.gz", cache_dir)


def resolve_library_path(lib_path=None, cache_dir=None) -> Path:
    """Use an explicit library or download a pinned binary for this platform.

    System dependencies are not installed; the caller loads the returned file.
    """
    if lib_path is not None:
        return _local_path(lib_path)
    system, machine = platform.system(), platform.machine().lower()
    machine = {"aarch64": "arm64", "amd64": "x86_64"}.get(machine, machine)
    asset = NATIVE_LIBRARIES.get((system, machine))
    if asset is None or struct.calcsize("P") != 8:
        raise RuntimeError(
            f"No prebuilt Hush library for {system}/{machine} with this Python architecture; "
            "provide lib_path or select backend='python'"
        )
    filename, checksum = asset
    name = Path(filename)
    return _download_asset(
        NATIVE_BASE_URL + filename, checksum, f"{name.stem}-{checksum}{name.suffix}", cache_dir,
    )


def _local_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Hush asset not found: {path}")
    return path


def _download_asset(url, checksum, filename, cache_dir):
    cache = (Path(cache_dir).expanduser() if cache_dir is not None
             else Path.home() / ".cache" / "aiavatar" / "hush")
    cache = cache.resolve()
    path = cache / filename
    if path.is_file() and _sha256(path) == checksum:
        return path
    cache.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, prefix=".hush-", suffix=".tmp", delete=False) as output:
            temporary_path = Path(output.name)
            digest = hashlib.sha256()
            with httpx.stream("GET", url, follow_redirects=True,
                              timeout=httpx.Timeout(60.0, connect=10.0)) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != checksum:
                raise ValueError(f"Downloaded Hush asset failed SHA-256 verification: {filename}")
        try:
            temporary_path.replace(path)
        except OSError:
            # Another process may have published and loaded the DLL on Windows.
            if not path.is_file() or _sha256(path) != checksum:
                raise
        return path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
