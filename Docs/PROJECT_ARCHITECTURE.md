# Runtime architecture and maintenance map

This guide describes the current execution path, rather than the historical
experiments in the phase reports. Start with the [README](../README.md) for
installation, [PatchCore explanation](PATCHCORE_EXPLANATION.md) for the algorithm,
and [API reference](API_DOCUMENTATION.md) for HTTP integration.

## Entry points

| Entry point | Responsibility |
| --- | --- |
| `app/app.py` | Streamlit upload, category selection, inspection display and report downloads |
| `app/backend.py` | FastAPI routes, upload validation, category routing, lazy detector cache and response assembly |
| `inference.py` | CLI dispatch to the selected detector; defaults to `patchcore_v23` |
| `src/detection/patchcore_v23_detector.py` | Production preprocessing, scoring, localization and visualization |
| `src/models/patchcore_v22.py` | Shared feature extractor, nearest-neighbour search and memory-bank loading used by v2.3 |
| `src/detection/category_checker.py` | Category prototype comparison and mismatch warnings |
| `src/utils/config.py` | YAML loading and category configuration resolution |
| `app/utils/pdf_report.py` | Report generation from inspection results |

The v2.3 detector deliberately imports `PatchCoreModelV22`; its filename does not
mean the CLI selected the legacy v2.2 detector.

## Request flow

```text
Streamlit image + chosen category
  -> POST /predict on configured backend
     -> if connection fails: session switches to in-process backend helpers
  -> get_detector(category), get_compatibility_checker()
  -> validate/decode upload -> detector.inspect(PIL image)
  -> category compatibility check -> assemble response
  -> display verdict, regions, heatmap, explanation -> optional PDF
```

The frontend defaults to `http://127.0.0.1:8000`, configurable with
`VISIONINSPECT_BACKEND_URL`. A network exception enables its in-process fallback;
an HTTP error is displayed as an inspection error. FastAPI is therefore part of
the local service path, but a separately running HTTP server is unnecessary for
the in-process fallback. The helpers in `app/backend.py` are shared by both paths.
The CLI calls detectors directly, bypassing HTTP upload validation and the
backend compatibility response wrapper.

## Detection and output

The detector converts images to RGB, resizes to **256 x 256**, and converts them
to tensors. ResNet18 layers 1, 2 and 3 form 448-dimensional vectors on a 64 x 64
grid. Each vector is compared with the selected category's normal memory bank.
Patch distances are interpolated into an anomaly map, optionally corrected by a
spatial prior, and smoothed. Category-specific aggregation produces the image
score. Pixel thresholding, morphology, component filtering, border exclusion and
region merging localize defects. Regions are mapped back to the original image.

The ordinary verdict is `score > image_threshold`; the detector can recover a
region when initial localization filters remove all candidates. Transistor also
has configured canonicalization and a missing-component presence override.
Region count alone is therefore not a universal verdict rule. Compatibility
checking follows detection in the backend and produces a warning without
automatically selecting another category.

## Artifacts and configuration

Each supported category (`bottle`, `leather`, `transistor`, `zipper`, `screw`) has
`models/<category>/patchcore_v23/` containing a memory bank, metadata and available
spatial priors. Category prototypes are stored separately. The backend normalizes
category names and rejects names outside its allowlist before constructing a
detector. Its directory check confirms directory existence, not checkpoint
integrity; use `python scripts/verify_models.py` for artifact checks.

`get_category_config()` takes explicit values from `categories` first, fills
missing keys from `phase3_2_categories`, then applies base/default values.
The detector resolves the prior mode and loads metadata. Do not interpret a
historical report's threshold table as a runtime configuration override.

The backend retains one active category detector, reuses it for repeated requests,
and purges the cache when categories change. The compatibility checker is lazy
and separate. These are process-local objects; multiple server workers have
separate state. `/health` reports loaded cache keys, and does not run inference or
prove that every checkpoint is usable.

## Safe contribution boundaries

Documentation and tests can improve understanding and regression coverage without
changing the model. Keep checkpoint tensors, prototypes, metadata, thresholds,
feature extraction, scoring, frontend behavior and deployment configuration out
of documentation/testing changes. See [CONTRIBUTING](../CONTRIBUTING.md) and the
[testing guide](TESTING_GUIDE.md).
