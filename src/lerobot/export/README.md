# lerobot export

`lerobot-export` is the command-line tool for exporting trained lerobot VLA
policies to C++-loadable artifacts. It lives entirely inside the `lerobot`
package (no fork, no separate install) and calls only public policy APIs.

---

## Two-tier install

### Tier 1: AOTI backend (vanilla AOTInductor .pt2)

Works on the **default lerobot install** (torch 2.11+ with no version changes).

```bash
pip install "lerobot[export]"
lerobot-export \
  --policy.path=lerobot/smolvla_base \
  --backend=aoti \
  --output_dir=bundles/smolvla_aoti
```

The output bundle contains `.pt2` files loadable by `torch::inductor::AOTIModelPackageLoader`
in C++, plus a `manifest.json` with all shapes and preprocessing recipes, plus
a reference C++ runner in `runner/`.

### Tier 2: torch-trt and torch-trt-et backends (needs torch 2.12)

torch-tensorrt 2.12 requires torch 2.12. Because the default lerobot install
uses the CUDA 12.8 PyTorch index (which has no torch 2.12 wheel), you need a
**separate export venv**:

```bash
python -m venv ~/lerobot-export-env
source ~/lerobot-export-env/bin/activate
pip install -r src/lerobot/export/requirements-export-trt.txt
pip install "lerobot[smolvla,export]"

# torch-trt AOTI (.pt2 with TRT engines):
lerobot-export \
  --policy.path=lerobot/smolvla_base \
  --backend=torch-trt \
  --output_dir=bundles/smolvla_trt

# torch-trt + ExecuTorch (.pte):
lerobot-export \
  --policy.path=lerobot/smolvla_base \
  --backend=torch-trt-et \
  --output_dir=bundles/smolvla_pte
```

The export venv is a build-time tool. The deployed artifact runs on the target
device using its own libtorch or ExecuTorch runtime, independent of this venv.

---

## Output bundle layout

```
bundles/smolvla_aoti/
  prefill.pt2           # Graph A: images + language + state -> KV cache
  denoise_step.pt2      # Graph B: noised actions + KV cache -> velocity
  manifest.json         # shapes, dtypes, preprocessing and decode recipes
  normalization.safetensors   # mean/std tensors for state/action
  runner/
    runner.cpp          # reference C++ runner (template, adapt for your robot)
    CMakeLists.txt      # builds against libtorch
  README.md
```

---

## C++ runner (for AOTI)

The reference runner in `runner/` shows how to:

1. Load `prefill.pt2` and `denoise_step.pt2` via `AOTIModelPackageLoader`.
2. Preprocess images and tokenize the instruction (host-side, per `manifest.json`).
3. Run the prefill pass once to get the KV cache.
4. Run the Euler decode loop (`n_steps` iterations) to get the action chunk.
5. Unnormalize and send the first action to the robot.

Build it against the libtorch that matches the torch version used for export:

```bash
cmake -B build \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DBUNDLE_DIR=/path/to/bundles/smolvla_aoti \
  src/lerobot/export/runner
cmake --build build
./build/lerobot_runner bundles/smolvla_aoti "pick up the block"
```

---

## Adding a new policy

Create `src/lerobot/export/adapters/<policy_name>.py`, define a class with
`policy_name = "<name>"`, and decorate it with `@register_adapter`.
The adapter must implement:

- `graphs(policy, cfg)`: return a dict mapping graph names to `(module, example_inputs)`.
  Compose only public policy attributes (`policy.model.*`). Do not edit model code.
- `manifest_extra(policy)`: return the decode recipe and preprocessing metadata.

See `src/lerobot/export/adapters/smolvla.py` for the SmolVLA reference.
