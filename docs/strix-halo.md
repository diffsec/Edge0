# Edge0-120B on Strix Halo

This branch runs the official Edge0-120B Q3_K pack on the Radeon 8060S
(`gfx1151`) through ROCm/HIP. The validated host is a Ryzen AI Max+ 395 with
128 GB unified memory, CachyOS, and ROCm 10.2.0 from `/opt/rocm/current`.

## Build and run

The build script does not install or replace system packages. It pins the
llama.cpp source archive, builds native `gfx1151` code, installs the matching
hipBLASLt `100500` tuning table, and creates the local Python environment:

```sh
scripts/build-edge0-strix-halo.sh
```

Keep the official model folder unchanged at the default location:

```text
/mnt/ai-models/edge0/Edge0-120B-q3k
```

Start the server on loopback port 8085:

```sh
scripts/run-edge0-120b.sh
```

The default context is 32,768 tokens. These environment variables provide
local overrides without editing source:

```sh
EDGE0_MODEL_DIR=/another/model/path \
EDGE0_CONTEXT=16384 \
EDGE0_PORT=8085 \
scripts/run-edge0-120b.sh
```

Check the API:

```sh
curl http://127.0.0.1:8085/v1/models

curl http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"edge0-120b","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Validated configuration

- Native `gfx1151`, wave32, WMMA, Q3_K expert, and hipBLASLt paths.
- Q8 latent dense projections (`latent_dense_q8.bin`).
- All 24,576 routed experts resident in unified GPU memory.
- Q3_K fused kernels and MTP speculative decoding remain enabled.
- PLE uses direct I/O instead of loading its 51.2 GB table into RAM.
- The engine keeps 32,768 KV cells resident at the 32K default.
- The server advertises text input and text output only; vision is unsupported.

Focused HIP parity and smoke tests pass on the physical `gfx1151` device.
OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints pass.

Measured validation results from 2026-11-15:

| Prompt | Cache | Prefill | Decode | Result |
| --- | ---: | ---: | ---: | --- |
| 63 tokens | cold | 52.3 tok/s | 37.8 tok/s | deterministic smoke passed |
| 7,068 tokens | cold | 720.1 tok/s | 34.0 tok/s | prefill passed |
| 16,018 tokens | cold | 692.3 tok/s | 32.2 tok/s | retrieval passed |
| 28,567 tokens | cold | 670.7 tok/s | 34.7 tok/s | retrieval passed |
| 32,067 tokens | 16,384 reused | 658.8 tok/s | 32.6 tok/s | retrieval passed |

These are correctness-run measurements, not a controlled benchmark suite.

## Required workarounds and current limit

`STRATA_VERIFY_ALL_RESIDENT=0` is required. The zero-doorbell verification
path stalls its first window when every expert is resident on this APU. The
ordinary GPU verification path remains enabled.

The parked conversation cache is disabled with
`--conversation-cache-mib 0`. With the default 8 GB cache, the first request
passes, but a later fused Q3_K prefill faults in `latent_w11_kernel<true>`.
Live prefix reuse still works; a repeated 7K request reused 7,063 tokens.

Contexts through 32,067 prompt tokens are validated. A 64K configuration
loads successfully, but a cold 57,669-token request faults in the same fused
Q3_K kernel after processing 57,344 tokens. The watchdog returns HTTP 503.
Memory capacity was not exhausted. Treat contexts above 32K as unsupported on
this branch until the kernel lifecycle fault is fixed and retested.

The runtime may produce a very large `gpucore.<pid>.gpu` file after an HSA
fault. These files are ignored by Git and can be removed after diagnostics are
captured.
