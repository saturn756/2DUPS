# 2DUPS agent / collaborator entry point

This file is the operational instruction for work in this repository. Read it before editing. Do not treat archived research, diagrams, placeholder implementations, or generated files as a second source of truth.

## Authoritative order

1. `docs/architecture/02_模块规范.md`: project scope, M0–M5 responsibilities, dataflow and degradation.
2. `docs/interface/04_接口规范.md` together with `src/twodups/contracts/`: I1–I9 wire fields. Code and examples must agree; `to_wire()` emits JSON payloads.
3. `docs/team/01_协作开发与复现规范.md`: environment, dataset, checkpoints and the minimal collaboration checks.
4. `configs/*.yaml` and `configs/dataset_manifest.yaml`: current values and sample identity. `docs/datasets/08_BDD100K_五段小样本.md` gives the exact acquisition commands.

`README.md` is the quick start. `docs/reference/` is historical investigation, **not** a current requirement. If sources disagree, do not pick one silently: fix code, its contract test and the relevant authoritative doc in the same change.

Collaborators use separate servers and checkouts. Never assume another member's hostname, account, Conda prefix, dataset directory or checkpoint directory exists locally. All committed paths and CLI examples must be repository-relative. Share Git commits, exact configs and verified hashes; each server obtains its own ignored data and weights.

## Implementation status and safe entry points

| Part | Status | Run command / implementation location |
|---|---|---|
| BDD100K input (I1/I2) | Working for 5-video subset only | `scripts/run_dataloader.py`; `src/twodups/data/` |
| M0 data ingest (I1/I2) | BDD100K five-video adapter works; other adapters are placeholders | `scripts/run_dataloader.py`; `src/twodups/data/` |
| M1 preprocessing (I3) | Initial quality gate, denoising and CLAHE enhancement; thresholds uncalibrated | `scripts/run_m1_preprocess.py`; `src/twodups/modules/m1_preprocess/` |
| M2 detection (I5) | Official YOLO11n weight hash pinned; BDD100K → M1 → GPU inference verified on one server, not accuracy-evaluated | `scripts/run_m2_detection.py --sample-id 359ce11c-e2f58b33 --max-frames 3`; `src/twodups/modules/m2_detection/` |
| M3 segmentation; M5 integration; E evaluation | Planned / `NotImplementedError` | Do not report as working; implement one module at a time |
| M4 tracking (I6) | Minimal same-class greedy IoU tracking-by-detection works without geometry compensation | `src/twodups/modules/m4_tracking/`; focused tests |

The current in-process M0→M1 handoff is `LoadedFrame(I2 metadata + RGB ndarray)`; M1→M2 uses I3 plus the RGB array for `default_variant`. Arrays are runtime carriers, not serializable contracts. Between processes, serialize the contract and dereference media deliberately. The verified runner currently covers M0→M1→M2 detection; it does not claim a full M0→M5 run. `run_pipeline.py --dry-run` only validates configuration loading. The Linux lock includes CUDA 12.8 PyTorch and Ultralytics, not any YOLO checkpoint; obtain the official weight separately as described in the team guide. M3 and M5 are still unimplemented; adjacent frames with differing default variants must not reuse an I4 transform.

## Where to edit

| Change | Edit together |
|---|---|
| A contract field, status, relation, alignment rule | `src/twodups/contracts/`, `docs/interface/04_接口规范.md`, contract/example tests; architecture if edge semantics changed |
| A module algorithm or threshold | `src/twodups/modules/<module>/`, `configs/m<id>_*.yaml`, focused tests; update architecture only if responsibilities change |
| Dataset selection or paths | `configs/dataset_manifest.yaml`, `configs/datasets/*.sha256`, acquisition documentation and reproducibility test |
| Dependency | `requirements.txt`, platform lock file, `environment.yml` when Python changes; test in isolated `2dups` environment |
| Checkpoint/model | When used, commit its repository-relative path and expected SHA-256 in the module config; never Git-add weights |
| Working entry point | `scripts/`, `README.md`, run smoke test |

Do not modify `docs/reference/` to encode current operational rules. The SVG linked from `README.md` is the current main architecture diagram and must change when the module topology changes; the PNG beside it is a non-authoritative export. Do not add a new README per module; the three authoritative docs and config/code comments are the maintainable documentation set.

## Required checks before sharing a change

From repository root in the project-specific Linux environment:

```bash
conda activate 2dups
python -m pip check
python -m pytest -q
python scripts/check_contracts.py
git diff --check
git status --short
```

For data-dependent changes also run `sha256sum -c configs/datasets/bdd100k-five.sha256`, then the relevant `scripts/run_dataloader.py`, `scripts/run_m1_preprocess.py` or `scripts/run_m2_detection.py` smoke command. Tests must not require private/raw data; data checks are separate. Before commit inspect `git diff --cached --name-only` for raw data, outputs, checkpoints, credentials or accidental generated files. The shared baseline is one Git commit (including configs), the verified dataset files and, when used, the verified checkpoint. Do not claim detection accuracy, M4 tracking metrics, M3/M5/E completion or a full M0–M5 end-to-end run passed while those entries remain unverified or placeholders.

## Robustness rule

Be strict about required fields, frame/sequence/variant/coordinate alignment, invalid geometry, file hashes, calibration and failure states. Accept only documented optional inputs; a future wire parser may ignore unknown optional fields, but the current repo has no such parser. Do not turn a missing input into `ok`, recycle a previous frame as a current detection, fabricate `rectified`, metric 3D/risk information, or use future frames. Keep the raw image and a reasoned fallback when an optional M2 transform fails.
