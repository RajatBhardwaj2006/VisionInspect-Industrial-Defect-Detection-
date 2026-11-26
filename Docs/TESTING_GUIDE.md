# Testing VisionInspect

Run commands from the repository root. Use Python 3.12 (the project target),
an isolated virtual environment and the existing dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux activate with `source .venv/bin/activate`. `pytest`, `httpx` and
`python-multipart` are already in `requirements.txt`; no separate test dependency
file or API keys are needed. Consult the [README](../README.md) for CPU PyTorch
installation and Git LFS setup. A fresh install needs network access for packages.

## Test commands

```powershell
# Complete maintained regression suite
python -m pytest tests/ -v

# New model-independent API/cache coverage
python -m pytest tests/test_api_errors.py tests/test_backend_routing.py -v

# One file, one test, or a keyword selection
python -m pytest tests/test_api_errors.py -v
python -m pytest tests/test_backend_routing.py::test_switching_category_evicts_previous_detector -v
python -m pytest tests/ -k missing_model_directory -v
```

`python -m pytest` without a path also discovers files named `test_*.py` in
`scripts/`; those include historical evaluation utilities. In particular,
`scripts/test_patchcore_v22.py` and `tests/test_patchcore_v22.py` share a module
name and cause an import-file-mismatch collection error with pytest's default
import mode. Use the maintained `tests/` command above. Record the exact command
when reporting results. Do not run parallel workers for the existing
integration suite: some tests use shared detector caches and the API rate limiter.

On Windows, if the numerical libraries report duplicate OpenMP initialization,
the project's documented local workaround is
`$env:KMP_DUPLICATE_LIB_OK="TRUE"` before starting Python.

## Which tests need model artifacts?

| Scope | Requirements |
| --- | --- |
| `test_api_errors.py`, `test_backend_routing.py` | Installed imports; generated images, temporary directories and mocked inference/constructor; no checkpoints or dataset |
| Existing API/inference/category compatibility integration tests | Production banks/prototypes plus the ImageNet ResNet18 checkpoint used by torchvision |
| Dataset loader integration tests | Extracted MVTec data; skipped when the raw dataset is absent |
| Legacy autoencoder/v2.2 tests | Some skip when retired checkpoints are absent; inspect each skip reason |

The full suite is **not guaranteed offline on a fresh machine**. Creating the
feature extractor calls `resnet18(weights=ResNet18_Weights.DEFAULT)`, which may
download weights if they are not cached under torch's hub checkpoint directory
(affected by `TORCH_HOME`). Do not download or regenerate weights merely to
validate a documentation/testing contribution. Run the new isolated files, and
report full-suite prerequisites that are missing.

To attempt the full suite while explicitly blocking uncached torchvision weight
downloads, this Python wrapper leaves production code unchanged:

```python
import os
from pathlib import Path
import subprocess
import sys
import tempfile

# PYTHONPATH propagates to the CLI subprocess launched by test_model_dispatch.
with tempfile.TemporaryDirectory() as guard_dir:
    Path(guard_dir, "sitecustomize.py").write_text(
        'import torch.hub\n'
        'def refuse_download(*args, **kwargs):\n'
        '    raise RuntimeError("Uncached weight download blocked for validation")\n'
        'torch.hub.download_url_to_file = refuse_download\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = guard_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"], env=env,
    )
sys.exit(result.returncode)
```

The guard applies to pytest and inherited Python subprocesses. Cached checkpoints
are still used normally. Missing ones cause explicit failures;
this wrapper does not replace the model with random weights or claim integration
success. Some older model tests call `fit()` on small synthetic tensors and save
temporary banks; these are algorithm checks, not production retraining. Do not
run training or calibration scripts as test setup.

## Reading and reporting results

`PASSED` means the assertions ran successfully. `SKIPPED` means they did not run;
use `-rs` to read the reasons. `FAILED` means an assertion or runtime operation
failed; `ERROR` commonly means collection, import or fixture setup failed. Exit
code 0 indicates success, 1 test failures, 2 an interrupted/collection-error run,
and 5 no tests collected. A reported test count is evidence for that execution,
not a fixed project guarantee.

For failures, record the test name and traceback before changing anything. Check
dependency versions, working directory, LFS pointers, available artifacts and
rate-limiter state. Never remove assertions, alter thresholds or mock the whole
pipeline to turn an integration failure green. Report existing failures separately
from results for new tests. See [CONTRIBUTING](../CONTRIBUTING.md) for PR checks.
