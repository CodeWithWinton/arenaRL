# Contributing to ArenaRL

Thank you for your interest in contributing.

## Reporting Issues

Open an issue on [GitHub](https://github.com/CodeWithWinton/arenarl/issues) with:
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/ -v`
5. Run the linter: `ruff check src/`
6. Open a pull request

## Creating a New Environment

1. Add a new file in `src/arenarl/envs/`
2. Inherit from `BaseEnv` and implement `reset()`, `step()`, and `render()`
3. Register it in `src/arenarl/envs/__init__.py`
4. Add tests in `tests/`
5. Document it in `docs/environments.md`

## Development Setup

```bash
git clone https://github.com/CodeWithWinton/arenarl.git
cd arenarl
pip install -e ".[dev]"
pytest tests/ -v
```

## Code Style

- Linting: [ruff](https://github.com/astral-sh/ruff)
- Line length: 100 characters
- Follow PEP 8
- Add docstrings to all public classes and methods

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
