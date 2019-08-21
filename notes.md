
# HOWTO

activate the environment with
```bash
source bin/activate
```

run tests with
```bash
pytest -vv
```

# Assumptions

My major assumptions:

- All input arrays are such that `len(A.shape) == 2`.
- Arrays can be stored in memory.
- Input files are at least of the same kind as the samples provided. e.g. no missing values need to be handled, imported arrays do not need type conversions, etc.
