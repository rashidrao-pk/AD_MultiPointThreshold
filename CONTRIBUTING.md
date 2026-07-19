# Contributing

Thank you for your interest in contributing.

This repository contains research and development code for real-time anomaly detection in collaborative robotic environments using deep generative models.

We welcome:
- Bug reports
- Feature requests
- Documentation improvements
- Code optimizations
- Research extensions
- New anomaly scoring methods
- Explainability/XAI integrations
- Benchmarking contributions

---

# Development Setup

## 1. Clone Repository

```bash
git clone https://github.com/rashidrao-pk/AD_MultiPointThreshold
cd AD_MultiPointThreshold
```

## 2. Create Environment

```bash
conda create -n dm_unito python=3.9.18
conda activate dm_unito
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Contribution Workflow
Create a Branch
```bash
git checkout -b feature/my-new-feature
```

Examples:
```text
feature/new-anomaly-score
fix/inference-memory-bug
docs/readme-update
```

## Coding Guidelines
**Python Style**

- Follow PEP8 conventions
- Use meaningful variable names
- Keep functions modular
- Add comments where necessary

**Deep Learning Code**

- Keep training reproducible
- Save random seeds when possible
- Avoid hardcoded dataset paths
- Make GPU/CPU execution configurable

## Commit Messages

Use clear commit messages.

Examples:

```text
git commit -m "Add SSIM anomaly score"
git commit -m "Fix CUDA memory issue during inference"
git commit -m "Improve threshold calibration pipeline"
```

## Pull Requests

**Before submitting a PR:**

- Ensure code runs correctly
- Verify training/inference scripts
- Update documentation if needed
- Include examples/screenshots when relevant


## Reporting Issues

**Please include:**

- Operating system
- Python version
- CUDA version
- GPU information
- Full traceback/error logs
- Steps to reproduce

## Research Contributions

If you extend this work academically:

- Clearly describe your modifications
- Provide evaluation metrics
- Include dataset details
- Reference related publications

## Citation

If you use this repository in your research, please cite the related publications listed in:

- README.md
- CITATION.cff
