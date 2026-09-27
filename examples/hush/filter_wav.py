"""Apply Hush to a 16 kHz mono, 16-bit PCM WAV without using a microphone.

Run from the repository root with ``python -m examples.hush.filter_wav``.
Uses the official native library by default; --backend python selects the Python fallback.
Both backends use the same ONNX model weights.
See documents/vad-filter-hush.md for setup and model conversion.
"""

import argparse
import asyncio
from pathlib import Path
import time
import wave

from aiavatar.sts.vad.filters.hush import HushAudioFilter


async def enhance(source, destination, *, lib_path, model_path, atten_lim_db, cache_dir=None, backend="native"):
    async with HushAudioFilter(
        backend=backend, lib_path=lib_path, model_path=model_path,
        atten_lim_db=atten_lim_db, cache_dir=cache_dir,
    ) as hush:
        while samples := await asyncio.to_thread(source.readframes, 512):
            output = await hush.process_async(samples, "wav")
            await asyncio.to_thread(destination.writeframesraw, output)
        tail = await hush.flush_async("wav")
        await asyncio.to_thread(destination.writeframesraw, tail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "python"), default="native",
                        help="Implementation: official native library or Python fallback (default: native)")
    parser.add_argument(
        "--lib", type=Path,
        help="Local libweya_nc library; omitted: download the supported native binary; invalid with python",
    )
    parser.add_argument(
        "--model", type=Path,
        help="Local Hush tar.gz (either backend) or converted ONNX (--backend python only); omitted: download the official bundle",
    )
    parser.add_argument("--cache-dir", type=Path, help="Official library/model cache (default: ~/.cache/aiavatar/hush)")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New WAV path; existing files are never overwritten")
    parser.add_argument("--atten-lim-db", type=float, default=100.0)
    args = parser.parse_args()
    if args.backend == "python" and args.lib is not None:
        parser.error("--lib cannot be used with --backend python")
    started = time.perf_counter()
    with wave.open(str(args.input), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (16000, 1, 2, "NONE"):
            parser.error("Input must be an uncompressed 16000 Hz mono, 16-bit PCM WAV")
        duration = source.getnframes() / source.getframerate()
        with args.output.open("xb") as output_file, wave.open(output_file, "wb") as destination:
            destination.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            asyncio.run(enhance(
                source, destination, lib_path=args.lib, model_path=args.model,
                atten_lim_db=args.atten_lim_db, cache_dir=args.cache_dir, backend=args.backend,
            ))
    elapsed = time.perf_counter() - started
    print(f"Saved {duration:.2f}s of audio to {args.output} ({elapsed:.2f}s including model loading)")


if __name__ == "__main__":
    main()
