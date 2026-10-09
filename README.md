# package_name

A template for personal Python projects, using [uv](https://docs.astral.sh/uv/) for dependency
management, [ruff](https://docs.astral.sh/ruff/) for linting/formatting, and
[ty](https://docs.astral.sh/ty/) for type checking.

## Using this template

1. Rename `src/package_name` to your project's package name and update `pyproject.toml`
   (`[project].name`, `[tool.uv.build-backend].module-name`) and `tests/unit/test_example.py`
   to match.
2. Update this README.
3. Run `uv sync --all-groups` to create the virtual environment and lock file.

## Commands

```
uv run pytest              # run tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ty check            # type check
```
