"""Compile either backend into a shared library loaded by ctypes."""

import argparse
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def library_path(backend: str) -> Path:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    if backend == "metal":
        return ROOT / "build" / "libkv_attention_metal.dylib"
    return ROOT / "build" / f"libqkv_{backend}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "metal"), default="cpu")
    args = parser.parse_args()
    system = platform.system()
    if system not in ("Darwin", "Linux"):
        parser.error("This demo's build script supports macOS and Linux.")
    if args.backend == "cuda" and system != "Linux":
        parser.error("The CUDA backend requires Linux and an NVIDIA CUDA toolkit.")
    if args.backend == "metal" and system != "Darwin":
        parser.error("The Metal attention backend requires macOS.")

    compiler = (
        shutil.which("nvcc") if args.backend == "cuda"
        else shutil.which("clang++") or shutil.which("g++")
    )
    if compiler is None:
        parser.error("Missing compiler: install nvcc for CUDA or clang++/g++ for CPU.")

    destination = library_path(args.backend)
    destination.parent.mkdir(exist_ok=True)
    if args.backend == "metal":
        command = [compiler, "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                   "-framework", "Foundation", "-framework", "Metal",
                   str(ROOT / "native" / "kv_attention_metal.mm")]
    elif args.backend == "cuda":
        command = [compiler, "-std=c++17", "-O2", "--fmad=false", "-shared",
                   "-Xcompiler", "-fPIC", str(ROOT / "native" / "qkv.cu")]
    else:
        shared_flag = "-dynamiclib" if system == "Darwin" else "-shared"
        command = [compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-fPIC",
                   shared_flag, str(ROOT / "native" / "qkv.cpp")]
    command.extend(["-o", str(destination)])
    print("Building:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print("Built:", destination)


if __name__ == "__main__":
    main()
