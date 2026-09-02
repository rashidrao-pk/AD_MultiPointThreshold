# Preparing a Commit

This project uses [`pre-commit`](https://pre-commit.com/) to keep Python code and
configuration files consistent before changes are committed. The configured
hooks format and lint Python code with Ruff, validate YAML and TOML files, remove
trailing whitespace, ensure files end with a newline, and detect unresolved merge
conflict markers.

## One-Time Setup

Install the development dependencies, then enable the Git hooks for this clone:

```bash
pip install -e ".[dev]"
pre-commit install
```

After installation, the checks run automatically whenever you create a commit.

## Check Your Changes

Before committing, run all configured checks across the repository:

```bash
pre-commit run --all-files
```

Some hooks, such as Ruff formatting and automatic lint fixes, may update files.
If that happens, review and stage the corrected files, then run the command again
until every check passes:

```bash
git diff
git add <files>
pre-commit run --all-files
```

You can also run the checks only on files that are already staged:

```bash
pre-commit run
```

## Create the Commit

Once the checks pass, commit the staged changes with a short, specific message
that describes the result of the change:

```bash
git commit -m "Improve threshold calibration pipeline"
```

Good commit messages use an imperative verb and focus on one logical change. For
example, `Add SSIM anomaly scoring`, `Fix CUDA memory usage during inference`, or
`Document local dataset configuration`.

If a check fails, fix the reported issue and commit again. Avoid bypassing the
hooks unless there is a documented and reviewed reason to do so.


```bash
git add .
pre-commit run --all-files
git add .
git commit -m "Improve threshold calibration pipeline"
```
