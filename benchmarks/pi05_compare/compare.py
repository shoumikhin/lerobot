"""Action error of one path against a float32 reference: same frames, same noise.

    python compare.py REFERENCE.npz TEST.npz [NAME]

Both files hold `actions` shaped (frames, 10, 7), in robot units. Prints the largest and the mean absolute error
per action value, and the cosine similarity of each frame's (10, 7) chunk (mean and smallest over the frames).
"""

import json
import sys

import numpy as np

reference = np.load(sys.argv[1])["actions"].astype(np.float64)
test = np.load(sys.argv[2])["actions"].astype(np.float64)
assert reference.shape == test.shape, (reference.shape, test.shape)
diff = np.abs(reference - test)
cosine = [
    float(r.ravel() @ t.ravel() / (np.linalg.norm(r) * np.linalg.norm(t) + 1e-12))
    for r, t in zip(reference, test, strict=True)
]
print(
    "COMPARE "
    + json.dumps(
        {
            "name": sys.argv[3] if len(sys.argv) > 3 else sys.argv[2],
            "frames": len(reference),
            "max_abs_err": round(float(diff.max()), 5),
            "mean_abs_err": round(float(diff.mean()), 5),
            "cosine_mean": round(float(np.mean(cosine)), 6),
            "cosine_min": round(float(np.min(cosine)), 6),
            "reference_abs_mean": round(float(np.abs(reference).mean()), 5),
        }
    )
)
