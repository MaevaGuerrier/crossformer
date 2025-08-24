import numpy as np
import flax

# Load npz
loaded = np.load("converted_checkpoint.npz", allow_pickle=True)

# Example: reconstruct nested dict if needed
params = {k: loaded[k] for k in loaded.files}

# print(f"param shape {[v.shape for v in params.values()]}")
print(f"Loaded parameters: {list(params.keys())}")