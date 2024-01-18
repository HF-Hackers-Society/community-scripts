# Community Scripts

HuggingFace community scripts reworked, modified, etc.

- [Diffusers](https://github.com/huggingface/diffusers/tree/main/examples/community)
- [Transformers](https://github.com/huggingface/transformers/tree/main/examples)

## Setup

### Updating

```
conda-lock -k explicit --conda mamba
# Update Conda packages based on re-generated lock file
mamba update --file conda-linux-64.lock
# Update Poetry packages and re-generate poetry.lock
poetry update
```

### Why Anaconda?

The major disadvantage to Anaconda is that many times the packages on its repos are outdated compared to present Pip versions. This leads many developers to prefer Pipenv or Poetry. However, in this case, managing PyTorch against multiple platforms is a major pain if you've ever tried to manage it with Poetry. And Anaconda handles PyTorch seamlessly. For all Pip dependencies, Poetry handles the rest, which gives us the best of all worlds!

### Configuration

After completing the steps below for your respective platform, run the following:

```
# Be sure you're in the "base" env!
mamba create -n hf --file conda-win-64.lock
conda activate hf
poetry env use python
poetry install
accelerate config
```

#### Windows

Install the latest available NVIDIA driver, then run `nvidia-smi` and verify the supported CUDA version is equal to or greater than the highest requirement!

Install the following:

- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    - Check "Desktop development with C++!"
    - Be sure to select the latest respective Windows SDK version to match your OS.
- [NVIDIA CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-toolkit-archive)
    - Or the latest version supported by [PyTorch](https://pytorch.org/get-started/locally/)!
    - Select "Custom" install, and uncheck everything but the CUDA runtime, development, and documentation sections.
- [NVIDIA CUDANN 12.x](https://developer.nvidia.com/cudnn)
    - Will need an NVIDIA developer account!
    - Be sure to get the version that supports the correct toolkit.
    - Follow the official [install guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows) so CUDA Toolkit updates don't break CUDANN.
    - Make the target install path `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDANN\12.x`.

```
# Other popular buckets: versions, java
scoop bucket add extras
scoop install rust mambaforge
mamba update mamba --all
```

#### Linux

Install `build-essential`, `cudatoolkit`, the latest compatible NVIDIA driver, `libsndfile1-dev`, `libgl1`, and `mamba`!

### Accelerate Config Example

```yml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### Final Checks

```
# Should print the conda env path
poetry env info -p
pytest
```

## Reporting issues

Attach the log from running `accelerate env`!
