## Install Conda

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
