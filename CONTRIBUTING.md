# Contributing

Bug reports and pull requests are welcome. A useful bug report names the GPU, the CUDA and driver versions, the
kernel, and the syncsan report or crash, and says whether `SYNCSAN_SHARED=0` or `--sample` changes it.

Before sending a change to the tool, run the known-answer cases on a GPU of compute capability 9.0 or newer:

```
NVBIT_ROOT=/path/to/nvbit_release_x86_64 tests/run_tests.sh sm_90
```

or, without a local GPU, `modal run reproduce/modal_cases.py`. All 37 cases must pass. A change that adds a pattern
syncsan should catch, or should stop reporting, adds a racy case and a clean case to `tests/race_cases.cu` and
`tests/run_tests.sh`.
