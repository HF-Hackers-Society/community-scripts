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

### Windows 10

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
scoop install rust miniconda3
conda config --set channel_priority strict
conda update conda --all
conda create -n hf --file conda-win-64.lock
conda activate hf
mamba install -c conda-forge cudatoolkit=11.8.0 cudnn=8.8.0.121
poetry install --no-root
```

If there are still any errors encountered from managing CUDA 11.8, download Purfview's [CUDA 11.x binaries (v3)](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs) and place them at: `C:\Users\{user}\scoop\apps\miniconda3\current\envs\hf\bin`.

### Linux

*Install drivers and cudatoolkit...*

```
conda create --name hf --file conda-linux-64.lock
poetry install
```

## Test

```
python -c "import torch; print('PyTorch: {}'.format(torch.cuda.is_available()))"
python -c "import tensorflow as tf; print('Tensorflow: {}'.format(len(tf.config.list_physical_devices('GPU')) > 0))"
```
