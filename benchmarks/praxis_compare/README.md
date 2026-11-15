# Praxis-1 speed: exported bundle, Runway's code, LeRobot eager

Times one action chunk (raw camera frames in, DROID actions out) on the same real DROID frames and the same GPU, for the exported bundle without PyTorch, Runway's [praxis-1-oss](https://github.com/runwayml/praxis-1-oss), and LeRobot's eager policy. Each pass is a fresh process with 3 warm-up and 30 timed chunks and `OMP_NUM_THREADS=8`; the result is the median of the valid passes' medians (see below). You need a LeRobot environment with `lerobot[praxis,export,dataset]` and a bundle from `lerobot-export`, plus Runway's environment and weights as its README describes. Set `BUNDLE_PY`, `RUNWAY_PY` and `RUNWAY_REPO` (see `run_ab.sh`), then run each comparison into a new output directory, with every side it compares in the same run, since the sampler settings are checked across the sides of one run. `run_ab.sh` fails when a pass gets no idle GPU in 3 tries, when a side has fewer passes than asked or no valid pass, or when the sides ran different sampler settings.

```bash
python make_sequence.py seq.npz                            # real DROID frames from lerobot/droid_100
bash run_ab.sh <gpu> <bundle dir> seq.npz out 3            # bundle and Runway, alternating passes
SIDES="bundle runway runway_cr lerobot" bash run_ab.sh <gpu> <bundle dir> seq.npz out_all 3   # with LeRobot eager
```

A pass is invalid when its median is more than 1.25 times the median of the same side's other per-pass medians, or when a warm-up chunk after the first takes more than 10 s. Other work on a shared machine, or a first pass that fills cold compile caches, can stretch a pass like that. The headline leaves invalid passes out, and each side's line lists the valid passes' medians, so a side with fewer valid passes shows it. One valid pass is enough for the run to succeed. The 90th percentile is printed but never excludes a pass.

## Jetson AGX Thor and DGX Spark

Make `seq.npz` on an x86_64 machine and copy it over. Frames made on aarch64 differ from those made on x86_64, so the run would not time the same frames as runs on other machines. Export the bundle on the machine that runs it, since its TensorRT engines are built for that GPU.

- Jetson AGX Thor: Runway's defaults run as they are. `nvidia-smi pmon` may not report other processes' GPU use there, so the wait for an idle GPU and the rerun of a disturbed pass may miss other work. Run on an otherwise idle board. `SIDES="bundle runway runway_nv"` adds Runway's NVFP4 experts. Point `RUNWAY_CKPT_NV` at Runway's NVFP4 checkpoint, or leave it unset and Runway packs the bfloat16 checkpoint to NVFP4 when it loads (the same kernels, other weight values).
- DGX Spark: set `RUNWAY_PAIR_GEMM_STAGES=3`. Runway's default paired expert GEMM needs more shared memory than the GB10 GPU has. Runway's NVFP4 refuses the GB10, so compare `bundle runway runway_cr`.

```bash
SIDES="bundle runway runway_nv" bash run_ab.sh 0 <bundle dir> seq.npz out_thor 3                    # Jetson AGX Thor
RUNWAY_PAIR_GEMM_STAGES=3 SIDES="bundle runway runway_cr" bash run_ab.sh 0 <bundle dir> seq.npz out_spark 3   # DGX Spark
```
