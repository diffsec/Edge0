# Community benchmark on AMD Radeon 8060S Graphics

Measured on 2026-11-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon 8060S Graphics, unified memory (the probe reports 0 GiB dedicated VRAM)
- CPU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (16 cores, 32 threads); installed RAM: 124.9 GiB; storage: zram0; nvme1n1 YMTC PC41Q-2TB-B       nvme; nvme0n1 Lexar SSD ARES PRO 2TB nvme
- OS: Linux 7.4.0-rc2-1-cachyos-rc; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.55, commit not a git checkout; engine 0.1.55
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: edge0-120b
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 28.0 (27.3-40.4) | 0.91 (0.81-13.71) |
| 4K prompt, fresh | 3939 (3939-3939) | 0 (0-0) | 24 cap | 3 | 666.0 (652.6-666.3) | not measured | 5.96 (5.96-6.09) |
| 16K prompt, fresh | 15716 (15716-15716) | 0 (0-0) | 24 cap | 3 | 644.7 (643.6-644.9) | not measured | 24.44 (24.44-24.48) |
| agent session, follow-up turns | 14406 (14401-14409) at the last turn | see below | up to 64 | 3 | see below | 27.6 (23.8-35.0) | 4.62 (2.56-7.12) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 65.4 % (65.4-65.6); earlier-turn tokens read again per session: 145 (73-145) (earlier turns that the cache did not hold; the 13515 (13515-13515) new tokens of the files are read once); first turn (nothing cached) first token: 1.36 s (1.35-1.39).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 459.4 | 1.392 | 33.5 |
| 1 | 2107 | 638 | 1469 | 563.9 | 2.679 | 27.6 |
| 2 | 5351 | 2100 | 3251 | 586.5 | 5.615 | 25.8 |
| 3 | 7420 | 5344 | 2076 | 555.1 | 3.803 | 30.3 |
| 4 | 10165 | 7482 | 2683 | 576.9 | 4.712 | 32.3 |
| 5 | 14409 | 10228 | 4181 | 592.2 | 7.125 | 24.3 |

### Skipped and failed

- skipped 32K: the context is 32768 tokens
- all measured requests passed, but the discarded calibration/warm-up phase
  caused three watchdog restarts after the engine stopped making progress at
  layer 17 of a small batched prompt; it recovered before measurements began

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. The warm-up restarts above mean this run demonstrates steady-state throughput, not clean-start stability. Total time of the measured suite: 390 s.
