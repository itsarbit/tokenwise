# Contributing to TokenWise

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/itsarbit/tokenwise.git
cd tokenwise
uv sync
```

## Running Tests

```bash
uv run pytest
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
```

## Making Changes

1. Fork the repo and create a branch from `master`
2. Make your changes
3. Add or update tests as needed
4. Run the full test suite and linter
5. Open a pull request

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting (100 char line length)
- Type hints are required — we run `mypy --strict`
- Keep changes focused — one concern per PR

## Reporting Issues

Open an issue at https://github.com/itsarbit/tokenwise/issues with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
