# Bounded run and data-comparison utilities

This optional Python package provides immutable run plans, executable and
input hashing, process limits, strict LHE3-subset checks, and seed-bootstrap
comparison utilities. It uses only the standard library. The simpler
packaged collision examples are driven by `benchmarks/run.py`.

From the repository root:

```sh
PYTHONPATH=tools/conference_harness python3 -m unittest discover   -s tools/conference_harness/tests
PYTHONPATH=tools/conference_harness python3 -m harness --help
```

A planned native run binds its exact card, binary, seed and backend request
in a token. Execution requires that token; arbitrary shell arguments are not
accepted. The optional `--make_gui_files` argument is supported by the v2 run
configuration. Process success and file integrity do not establish physical
accuracy. Synthetic LHE and analytic examples are labeled as such.
