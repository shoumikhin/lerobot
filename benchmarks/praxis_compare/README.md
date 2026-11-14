# Praxis-1 speed: exported bundle, Runway's code, LeRobot eager

Times one action chunk (raw camera frames in, DROID actions out) on the same real DROID frames and the same GPU, for the exported bundle without PyTorch, Runway's [praxis-1-oss](https://github.com/runwayml/praxis-1-oss), and LeRobot's eager policy. Each pass is a fresh process with 3 warm-up and 30 timed chunks and `OMP_NUM_THREADS=8`; the result is the median of 3 per-pass medians. You need a LeRobot environment with `lerobot[praxis,export,dataset]` and a bundle from `lerobot-export`, plus Runway's environment and weights as its README describes. Set `BUNDLE_PY`, `RUNWAY_PY` and `RUNWAY_REPO` (see `run_ab.sh`), then run each comparison into a new output directory, with every side it compares in the same run, since the sampler settings are checked across the sides of one run. `run_ab.sh` fails when a pass gets no idle GPU in 3 tries, when a side has fewer passes than asked, or when the sides ran different sampler settings.

```bash
python make_sequence.py seq.npz                            # real DROID frames from lerobot/droid_100
bash run_ab.sh <gpu> <bundle dir> seq.npz out 3            # bundle and Runway, alternating passes
SIDES="bundle runway runway_cr lerobot" bash run_ab.sh <gpu> <bundle dir> seq.npz out_all 3   # with LeRobot eager
```
