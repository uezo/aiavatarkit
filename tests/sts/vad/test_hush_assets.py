"""Hermetic asset-download checks: mocked HTTP and temporary caches only."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import threading

import httpx
import pytest

from aiavatar.sts.vad.filters.hush import assets


@pytest.fixture
def download(tmp_path, monkeypatch):
    content = b"synthetic Hush model archive"
    monkeypatch.setattr(assets, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
    requests = []
    handler = lambda request: httpx.Response(200, content=content)

    def dispatch(request):
        requests.append(request)
        return handler(request)

    with httpx.Client(transport=httpx.MockTransport(dispatch)) as client:
        def stream(method, url, **kwargs):
            assert kwargs["follow_redirects"] is True
            assert kwargs["timeout"].connect == 10.0
            assert kwargs["timeout"].read == 60.0
            return client.stream(method, url, **kwargs)

        monkeypatch.setattr(assets.httpx, "stream", stream)

        def set_handler(value):
            nonlocal handler
            handler = value

        yield tmp_path / "cache", content, requests, set_handler


@pytest.fixture
def native_download(download, monkeypatch):
    _, content, _, _ = download
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(assets, "NATIVE_LIBRARIES", {
        key: (filename, digest) for key, (filename, _) in assets.NATIVE_LIBRARIES.items()
    })
    monkeypatch.setattr(assets.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(assets.platform, "machine", lambda: "arm64")
    calcsize = assets.struct.calcsize
    monkeypatch.setattr(assets.struct, "calcsize", lambda fmt: 8 if fmt == "P" else calcsize(fmt))
    return download


def test_explicit_path_never_downloads_or_creates_cache(download):
    cache, content, requests, _ = download
    model = cache.parent / "custom.onnx"
    model.write_bytes(b"custom model, not the official bundle")
    assert assets.resolve_model_path(model, cache) == model.resolve()
    assert not cache.exists()
    assert requests == []


def test_explicit_library_skips_platform_detection_and_download(download, monkeypatch):
    cache, content, requests, _ = download
    library = cache.parent / "custom-library.so"
    library.write_bytes(b"user-provided native library")

    def reject_detection():
        pytest.fail("An explicit native library must not depend on automatic platform support")

    monkeypatch.setattr(assets.platform, "system", reject_detection)
    monkeypatch.setattr(assets.platform, "machine", reject_detection)
    assert assets.resolve_library_path(library, cache) == library.resolve()
    assert not cache.exists()
    assert requests == []


@pytest.mark.parametrize("directory", [False, True])
def test_invalid_explicit_library_fails_without_download(download, directory):
    cache, content, requests, _ = download
    library = cache.parent / "missing-library.so"
    if directory:
        library.mkdir()
    with pytest.raises(FileNotFoundError):
        assets.resolve_library_path(library, cache)
    assert not cache.exists()
    assert requests == []


@pytest.mark.parametrize("system,machine,normalized", [
    ("Darwin", "arm64", "arm64"), ("Darwin", "AARCH64", "arm64"),
    ("Linux", "x86_64", "x86_64"), ("Linux", "AMD64", "x86_64"),
    ("Windows", "AMD64", "x86_64"), ("Windows", "X86_64", "x86_64"),
])
def test_native_platform_selection_and_verified_cache_reuse(
    native_download, monkeypatch, system, machine, normalized,
):
    cache, content, requests, _ = native_download
    monkeypatch.setattr(assets.platform, "system", lambda: system)
    monkeypatch.setattr(assets.platform, "machine", lambda: machine)
    filename, digest = assets.NATIVE_LIBRARIES[(system, normalized)]
    name = Path(filename)
    path = assets.resolve_library_path(cache_dir=cache)
    assert path.name == f"{name.stem}-{digest}{name.suffix}"
    assert path.read_bytes() == content
    assert str(requests[0].url) == f"{assets.NATIVE_BASE_URL.rstrip('/')}/{filename}"
    assert assets.NATIVE_REVISION in str(requests[0].url)
    assert assets.resolve_library_path(cache_dir=cache) == path
    assert len(requests) == 1
    assert list(cache.iterdir()) == [path]


@pytest.mark.parametrize("system,machine,pointer_bytes", [
    ("Darwin", "x86_64", 8), ("Linux", "aarch64", 8),
    ("Windows", "ARM64", 8), ("FreeBSD", "x86_64", 8),
    ("Linux", "i686", 4), ("Darwin", "arm64", 4),
])
def test_unsupported_native_platform_fails_before_download(
    native_download, monkeypatch, system, machine, pointer_bytes,
):
    cache, content, requests, _ = native_download
    monkeypatch.setattr(assets.platform, "system", lambda: system)
    monkeypatch.setattr(assets.platform, "machine", lambda: machine)
    calcsize = assets.struct.calcsize
    monkeypatch.setattr(assets.struct, "calcsize", lambda fmt: pointer_bytes if fmt == "P" else calcsize(fmt))
    with pytest.raises(RuntimeError) as error:
        assets.resolve_library_path(cache_dir=cache)
    assert "lib_path" in str(error.value)
    assert "python" in str(error.value)
    assert not cache.exists()
    assert requests == []


@pytest.mark.parametrize("directory", [False, True])
def test_invalid_explicit_path_fails_without_download(download, directory):
    cache, content, requests, _ = download
    model = cache.parent / "missing.onnx"
    if directory:
        model.mkdir()
    with pytest.raises(FileNotFoundError, match="Hush asset not found"):
        assets.resolve_model_path(model, cache)
    assert not cache.exists()
    assert requests == []


def test_download_then_verified_cache_reuse(download):
    cache, content, requests, _ = download
    path = assets.resolve_model_path(cache_dir=cache)
    assert path.read_bytes() == content
    assert assets.MODEL_SHA256 in path.name
    assert str(requests[0].url) == assets.MODEL_URL
    assert assets.resolve_model_path(cache_dir=cache) == path
    assert len(requests) == 1
    assert list(cache.iterdir()) == [path]


def test_default_cache_uses_home_only_when_resolving(download, monkeypatch):
    cache, content, requests, _ = download
    monkeypatch.setattr(Path, "home", lambda: cache.parent)
    path = assets.resolve_model_path()
    assert path.parent == cache.parent / ".cache" / "aiavatar" / "hush"
    assert path.read_bytes() == content


@pytest.mark.parametrize("asset", ["model", "library"])
def test_corrupt_cache_is_replaced_and_other_files_preserved(native_download, asset):
    cache, content, requests, _ = native_download
    resolve = assets.resolve_model_path if asset == "model" else assets.resolve_library_path
    path = resolve(cache_dir=cache)
    path.write_bytes(b"corrupt")
    unrelated = cache / "user-file.txt"
    unrelated.write_text("preserve")
    assert resolve(cache_dir=cache) == path
    assert path.read_bytes() == content
    assert unrelated.read_text() == "preserve"
    assert len(requests) == 2
    assert set(cache.iterdir()) == {path, unrelated}


@pytest.mark.parametrize("failure", ["status", "checksum", "connection", "partial"])
@pytest.mark.parametrize("asset", ["model", "library"])
def test_failure_cleans_partial_and_preserves_existing_cache(native_download, failure, asset):
    cache, content, requests, set_handler = native_download
    resolve = assets.resolve_model_path if asset == "model" else assets.resolve_library_path
    path = resolve(cache_dir=cache)
    path.write_bytes(b"old corrupted cache")

    class InterruptedStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partially downloaded archive" * 100000
            raise httpx.ReadError("interrupted")

    def handler(request):
        if failure == "status":
            return httpx.Response(503)
        if failure == "checksum":
            return httpx.Response(200, content=b"unexpected archive")
        if failure == "connection":
            raise httpx.ConnectTimeout("timed out")
        return httpx.Response(200, stream=InterruptedStream())

    set_handler(handler)
    error = ValueError if failure == "checksum" else httpx.HTTPError
    with pytest.raises(error):
        resolve(cache_dir=cache)
    assert path.read_bytes() == b"old corrupted cache"
    assert list(cache.iterdir()) == [path]


@pytest.mark.parametrize("asset", ["model", "library"])
def test_concurrent_cold_downloads_publish_same_valid_file(native_download, asset):
    cache, content, requests, set_handler = native_download
    resolve = assets.resolve_model_path if asset == "model" else assets.resolve_library_path
    barrier = threading.Barrier(2)

    def handler(request):
        barrier.wait(timeout=5)
        return httpx.Response(200, content=content)

    set_handler(handler)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: resolve(cache_dir=cache), range(2)))
    assert results[0] == results[1]
    assert results[0].read_bytes() == content
    assert len(requests) == 2
    assert list(cache.iterdir()) == [results[0]]


@pytest.mark.parametrize("published", ["valid", "invalid", "missing"])
def test_native_replace_failure_reuses_only_verified_concurrent_publication(
    native_download, monkeypatch, published,
):
    cache, content, requests, _ = native_download
    monkeypatch.setattr(assets.platform, "system", lambda: "Windows")
    monkeypatch.setattr(assets.platform, "machine", lambda: "AMD64")
    failure = PermissionError("DLL replacement denied")
    targets = []

    def fail_replace(temporary, target):
        targets.append(target)
        assert temporary.parent == target.parent == cache
        assert target.suffix == ".dll"
        if published != "missing":
            target.write_bytes(content if published == "valid" else b"unverified library")
        raise failure

    monkeypatch.setattr(Path, "replace", fail_replace)
    if published == "valid":
        path = assets.resolve_library_path(cache_dir=cache)
        assert path.read_bytes() == content
    else:
        with pytest.raises(PermissionError) as error:
            assets.resolve_library_path(cache_dir=cache)
        assert error.value is failure
    assert len(requests) == len(targets) == 1
    assert list(cache.iterdir()) == ([] if published == "missing" else targets)
    if published == "invalid":
        assert targets[0].read_bytes() == b"unverified library"
