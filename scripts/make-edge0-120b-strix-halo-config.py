#!/usr/bin/env python3
"""Generate the local Edge0-120B HIP run config for a Strix Halo APU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rocm-root", type=Path, required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    model = args.model_dir.resolve()
    rocm = args.rocm_root.resolve()
    strata = repo / "edge0-strata"
    build = strata / "build-hip-gfx1151"
    exe = build / "strata"
    tuning = build / "tuning" / "gfx1151-hipblaslt-100500.txt"

    required = [
        exe,
        tuning,
        model / "model.json",
        model / "experts.bin",
        model / "labels.bin",
        model / "latent_dense_q8.bin",
        model / "model-skeleton.gguf",
        model / "ple-fp8.gguf",
        model / "profile.bin",
        model / "mtp",
        model / "tokenizer",
        strata / ".venv" / "bin" / "python",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required Edge0 files:\n  " + "\n  ".join(missing))
    if args.context < 4096 or args.context > 262144:
        raise SystemExit("--context must be between 4096 and 262144")

    def model_path(name: str) -> str:
        return str(model / name)

    config = {
        "exe": str(exe),
        "args": [
            "--pack", str(model),
            "--native", model_path("model-skeleton.gguf"),
            "--native-dense-gguf", model_path("model-skeleton.gguf"),
            "--ple-gguf", model_path("ple-fp8.gguf"),
            "--ple-io", "direct",
            "--expert-profile", model_path("profile.bin"),
            "--expert-cache", "auto",
            "--prefill", "auto",
            "--spec", "4",
            "--spec-min-p", "0.5",
            "--mtp", model_path("mtp"),
            "--max-context", str(args.context),
            "--kv", "int8",
            "--kv-resident", str(min(args.context, 32768)),
            # The current branch faults in a later fused prefill after parking a
            # conversation on gfx1151. Keep caching off until that lifecycle bug
            # is fixed; prefix reuse within a request is unaffected.
            "--conversation-cache-mib", "0",
            "--pool-workers", "15",
            "--vram-reserve-mib", "700",
        ],
        "cwd": str(strata),
        "tokenizer": model_path("tokenizer"),
        "model_name": "edge0-120b",
        "backend": "hip",
        "env": {
            "STRATA_LATENT_A": model_path("experts.bin"),
            "STRATA_LATENT_LABELS": model_path("labels.bin"),
            "STRATA_LATENT_DENSE": model_path("latent_dense_q8.bin"),
            "STRATA_LATENT_DENSE_FMT": "8",
            "STRATA_EMBED_VRAM": "1",
            "STRATA_LATENT_FUSED": "1",
            "STRATA_LATENT_FUSED_DU": "1",
            "STRATA_LATENT_U_GPU": "1",
            "STRATA_LAT_FUSED_Q3K": "1",
            # The zero-doorbell path hangs the first verify window when every
            # expert is resident on this APU. The ordinary GPU verifier works.
            "STRATA_VERIFY_ALL_RESIDENT": "0",
            "STRATA_HIPBLASLT_TUNING": str(tuning),
        },
        "lib_dirs": [
            str(rocm / "lib"),
            str(rocm / "lib" / "rocm_sysdeps" / "lib"),
            str(rocm / "lib" / "llvm" / "lib"),
        ],
        "gpus_asked": True,
        "log": str(build / "edge0-120b.log"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
