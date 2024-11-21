# Build instructions

## Create a conda environment

```
(base) conda create -n pyflexad-hatch python=3.11 -y
(base) conda activate pyflexad-hatch
(pyflexad-hatch) conda install -c conda-forge hatch -y
```

## (Optional) Create the package

```
(base) cd <repository root path>
(base) conda activate pyflexad-hatch
(pyflexad-hatch) hatch new --init
```

## Build the package

```
(base) cd <repository root path>
(base) conda activate pyflexad-hatch
(pyflexad-hatch) hatch env prune
(pyflexad-hatch) hatch env create
(pyflexad-hatch) hatch build
```

## (Optional) Test the package

```
(base) cd <repository root path>
(base) conda activate pyflexad-hatch
(pyflexad-hatch) hatch run test:pytest
```

## Publish the package

### (Optional) Publish the package on TestPyPI

```
(base) cd <repository root path>
(base) conda activate pyflexad-hatch
(pyflexad-hatch) hatch publish --repo test
```

### Publish the package on PyPI

```
(base) cd <repository root path>
(base) conda activate pyflexad-hatch
(pyflexad-hatch) hatch publish
```

## (Optional) Install the package

```
(base) conda create -n pyflexad-test python=3.11 -y
(base) conda activate pyflexad-test
```

### (Optional) Install the package from local wheel

```
(pyflexad-test) cd <repository root path>
(pyflexad-test) pip install .\dist\pyflexad-0.0.1-py3-none-any.whl
```

### (Optional) Install the package from TestPyPI

```
(pyflexad-test) pip install -i https://test.pypi.org/simple/ pyflexad
```

### (Optional) Install the package from PyPI

```
(pyflexad-test) pip install pyflexad
```

### (Optional) Test the installation

```
(pyflexad-test) python ./scripts/bess/script_pub_use_case.py
```
### (Optional) Export the test environment

```
(pyflexad-test) conda env export -n pyflexad-test --no-builds -f environment-test.yml
```