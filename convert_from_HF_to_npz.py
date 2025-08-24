import orbax.checkpoint
import jax
import numpy as np
from huggingface_hub import snapshot_download


# Paths
hf_checkpoint_path = "/home/mae/.cache/huggingface/hub/models--rail-berkeley--crossformer/snapshots/c7dea2691aed3656537c5126a0a77df84a28abd7" # Hugging Face downloaded checkpoint
output_npz_path = "converted_checkpoint.npz"
repo_id = "rail-berkeley/crossformer"
# local_dir = snapshot_download(repo_id=repo_id)
# print("Checkpoint downloaded to:", local_dir)



# Create checkpointer
checkpointer = orbax.checkpoint.CheckpointManager(
    hf_checkpoint_path,
    orbax.checkpoint.PyTreeCheckpointer()
)

# Get latest step
step = checkpointer.latest_step()

# Restore PyTree
params = checkpointer.restore(step)

# Flatten PyTree to dict of arrays
def flatten_pytree(pytree, prefix=""):
    flat_dict = {}
    if isinstance(pytree, dict):
        for k, v in pytree.items():
            flat_dict.update(flatten_pytree(v, f"{prefix}{k}/"))
    else:
        flat_dict[prefix[:-1]] = np.array(pytree)
    return flat_dict

flat_params = flatten_pytree(params)

# Save as npz
np.savez(output_npz_path, **flat_params)

print(f"Checkpoint converted to {output_npz_path}")
