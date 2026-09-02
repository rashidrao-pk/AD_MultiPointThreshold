## Install Conda

### PyTorch with cuda GPU on Windows

```bash
conda activate AD

python -m pip uninstall -y torch torchvision torchaudio

python -m pip install `
  torch==2.5.1 `
  torchvision==0.20.1 `
  torchaudio==2.5.1 `
  --index-url https://download.pytorch.org/whl/cu118

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

```


### Linux Setup
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
exec bash
```

Install `pre-commit-checks`

```bash
pre-commit install
```

```bash
pre-commit run --all-files
pre-commit run -a
```

## Use `prek`

```bash
pip install prek
```

```bash
prek run -a
```

```bash
prek run -a
pytest
git status
```

```bash
ruff check data/load.py
ruff format data/load.py
ruff check data docs notebooks
```

```bash
ruff check app data experiments models modules utils
ruff format app data experiments models modules utils

pytest
```
