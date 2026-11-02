# Tests

Configure a CMake build with `BUILD_TESTING=ON` (the default), then run:

```sh
ctest --test-dir build -LE strict --output-on-failure -j 1
ctest --test-dir build -L strict --output-on-failure -j 1
python -m unittest discover -s tools -p 'test_*.py'
python tools/check_browser_modules.py
```

The first command runs software regression targets. The second runs the
strict adaptive-field numerical qualification, which currently fails its
accuracy requirements. Report these separately.

The native tests cover field solvers, luminosity accumulation, sparse-cell
traversal, CPU/GPU execution, radiation, spin and secondary transport.
`examples/` also retains the upstream CLIC saved-output regression. Reference
output checks detect software changes; they are not independent collider
measurements. The [benchmark guide](../benchmarks/README.md) documents input
cards and quantitative reference comparisons.

GPU targets require a CUDA build and a compatible device. Commands must run
within the memory and CPU/GPU allocation available to the process.
