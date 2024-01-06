# Community Scripts

HuggingFace community scripts reworked, modified, etc.

- [Diffusers](https://github.com/huggingface/diffusers/tree/main/examples/community)
- [Transformers](https://github.com/huggingface/transformers/tree/main/examples)

## Setup

### Why Anaconda?

The major disadvantage to Anaconda is that many times the packages on its repos are outdated compared to present Pip versions. This leads many developers to prefer `Pipenv`. However, in this case, managing PyTorch against multiple platforms is a major pain if you've ever tried to manage it. And Anaconda handles PyTorch seemlessly. For all Pip dependencies, Poetry handles the rest, which gives us the best of all worlds.

### Windows

```
# Other popular buckets: versions, java
scoop bucket add extras
scoop install rust miniconda3
conda config --set channel_priority strict
conda update conda --all
conda create --name hf --file conda-win-64.lock
conda install -c conda-forge cudatoolkit=11.8.0 cudnn=8.8.0.121
poetry install
```

### Linux

*Install drivers and such...*

```
conda create --name hf --file conda-linux-64.lock
poetry install
```

## Test

```
python -c "import torch; print('PyTorch: {}'.format(torch.cuda.is_available()))"
python -c "import tensorflow as tf; print('Tensorflow: {}'.format(len(tf.config.list_physical_devices('GPU')) > 0))"
```
