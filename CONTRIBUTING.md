# Contributing to DeepTE-Peptide

Thank you for your interest in contributing to DeepTE-Peptide! We welcome contributions to models, preprocessing pipelines, documentation, and evaluation benchmarks.

## Development Workflow

1. **Clone the repository and set up environment**:
   ```bash
   git clone https://github.com/227182038-rgb/deepte-peptide.git
   cd deepte-peptide
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1
   pip install -e ".[dev]"
   ```

2. **Code style & quality**:
   - We format and lint with [Ruff](https://docs.astral.sh/ruff/):
     ```bash
     python -m ruff check src figures tests
     ```
   - Target line length is 100 characters.

3. **Running tests**:
   - Always run the test suite before submitting changes:
     ```bash
     python -m pytest tests -v --cov=deepte
     ```

4. **Pull Requests**:
   - Provide a clear description of changes.
   - Include unit tests for any new features or bug fixes.
   - Ensure all CI tests pass.
