# Releasing NanoSuperlattice

The package builds from `pyproject.toml` with setuptools. The distribution name is `nanosuperlattice`, and it installs two commands, `nanosuperlattice` (viewer) and `nanosuperlattice-integrations`.

## Before every release

1. Bump `__version__` in `nanosuperlattice/__init__.py`. PyPI and TestPyPI never accept the same version twice.
2. Run the tests: `python -m unittest discover -v tests`.

## Build and check locally

```sh
python -m pip install --upgrade build twine
python -m build
python -m twine check --strict dist/*
```

`dist/` then holds `nanosuperlattice-<version>.tar.gz` (source, including tests and examples) and `nanosuperlattice-<version>-py3-none-any.whl`.

## Publish to TestPyPI

**From GitHub (recommended, no token).** This uses the workflow `.github/workflows/publish-testpypi.yml`, which builds, tests the wheel and uploads it. You set it up once:

1. Sign in at https://test.pypi.org, open Your account → Publishing, and add a pending publisher with these values. Project `nanosuperlattice`, owner `nxuanloc2026`, repository `NanoSuperlattice`, workflow `publish-testpypi.yml`, environment `testpypi`.
2. In the GitHub repository, open Settings → Environments and create an environment named `testpypi`.

After that, each release starts from Actions → Publish to TestPyPI → Run workflow.

**From your computer.** Create an API token at https://test.pypi.org/manage/account/token/ and run:

```sh
python -m twine upload --repository testpypi dist/*
```

Use `__token__` as the user name and the token as the password.

**Check the upload** in a fresh environment:

```sh
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nanosuperlattice
nanosuperlattice-integrations --help
```

## Before the real PyPI

- Choose a license, add a `LICENSE` file and declare it in `pyproject.toml`. Without one, others have no legal right to reuse the code.
- Copy the workflow to `publish-pypi.yml` with the `repository-url` line removed and an environment named `pypi`, and register it as a trusted publisher on https://pypi.org.
