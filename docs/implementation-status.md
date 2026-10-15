# Implementation status

> 2026-10-15 refactor: ユーザー指定により今回の差分は未検証です。以前のPASS記録と区別し、[今回の実装引き継ぎ](refactoring-implementation-2026-10-15.md)を参照してください。

Plan: `Krea2_AFM_Codex_Implementation_Plan_2026-09-30.md`, revision 2.

## Current milestone

Tasks 2–4 (AFM editing) and offline Task 5 tooling are implemented in the requested
directory. The earlier decision to stop coding at the diagnostic GPU gate was
too narrow: the user excluded real-model testing, not the edit implementation.
Real-model evidence remains required for compatibility/performance claims and
auto-profile qualification, but no longer blocks implementation and CPU tests.

## Decisions and preflight

The diagnostic-stage scope rulings below are historical and superseded by the
edit continuation. No real-model testing has been performed in either stage.

- Task 0 → Task 1: immutable `AFMConfig`, native Krea2 capabilities and patch
  conflict detection feed runtime installation; interfaces agree.
- Task 1 → Tasks 2–4: layout and progress types retain the plan's interfaces.
- Ruling: work directly in the newly initialized repository on
  `diagnostic/krea2-afm`; there was no existing checkout to isolate. The
  requested output directory remains the installable package.
- Ruling: use the supplied design/plan and its explicit diagnostic milestone
  under the user's autonomous execution instruction. Further design approval
  is unnecessary. The historical GPU gate now applies to support and profile
  qualification claims; it does not block edit implementation or CPU tests.
- Ruling: absent a supplied server URL, use official upstream workflow/source
  evidence to prepare examples and a server-side validation handoff. Static
  validation is not a claim that cloud `/object_info` or inference passed.
- User clarification: real-model testing is excluded from this session. No
  model downloads, server submission, GPU inference, or cloud validation.
- License review: do not copy unlicensed Anima source. New implementation
  follows the supplied specification; third-party provenance is documented.

## Validation environment

- Windows; Python 3.10.11; PyTorch 2.10.0+cpu; pytest 9.0.2.
- No local ComfyUI import, CUDA runtime, or Krea2 inference environment.
- ComfyUI authoring skill bundle completeness: 37 checks passed.

## Tasks

- [x] Task 0: configuration, capabilities, conflict checks, tests, provenance.
- [x] Task 1 local implementation: diagnostic runtime, backend dispatch, nodes,
  examples and CPU tests. Live `/object_info` validation remains unperformed.
- [ ] Intermediate GPU gate: true baseline/discover/observe on Turbo and
  rectangle; confirm backend, grid, txtfusion exclusion, CFG, run isolation.
- [x] Task 2: shared FP32 FFT, normalized radial cosine masks, DC preservation.
- [x] Task 3: chunked joint manual attention and conservative workspace guard.
- [x] Task 3A: augmented Q/K/V kernel and fail-closed profile selection.
- [x] Task 4: edit node, zero controls, selected-row replacement, atomic fallback.
- [x] Task 5 offline tooling: numeric/report utilities, API/UI examples, dry-run matrix runner.
- [ ] Task 5 real-model/GPU/quality/performance evidence (explicitly excluded).

Test evidence and final review findings are appended during implementation.

## Evidence

- Task 0 RED: `tests/test_capabilities.py` failed collection because the new
  `afm` module did not exist. GREEN: 26 tests passed. Commit `0077c04`.
- Task 1 RED: runtime/backend tests failed collection for the missing modules;
  V3 node tests failed for the missing entry point. During GREEN, lifecycle
  testing exposed a duplicate `config` argument in JSON logging; the emitter
  parameter was renamed. All 69 capability/runtime/backend/node tests passed.
- API validator RED: missing validator module; GREEN: graph, choices, socket,
  missing-link and cycle tests passed.
- Full pre-review verification: 78 tests passed, including 3 source-only tests
  against upstream `wrap_attn` and `WrapperExecutor`. `compileall` and
  `git diff --check` passed. API graph validation passed without inference.
- Dedicated lint/typecheck tools are not installed; no new dependencies were
  installed. Syntax compilation and source-contract tests are the local checks.

The diagnostic-stage release exposed no edit settings. That release remains
in history at b81d70f; the current feat/krea2-afm-edit branch adds the edit implementation.

## Historical diagnostic review (b81d70f)

Independent code review identified five important issues. Each was reproduced
with a failing test before its fix:

1. Live patch inventory now rejects post-AFM post_input/QKV/output hooks before
   forward; only the exact AFM post_input hook is permitted.
2. Node log paths are resolved beneath `output/krea2_afm/`, with `.jsonl` suffix;
   absolute/traversal/resolved escapes are rejected. Resolved-escape tests
   simulate filesystem redirection; they do not create a privileged Windows symlink.
3. Malformed branch metadata is recorded as null with an unsupported reason;
   it cannot trigger JSON serialization failure after container consumption.
4. Summary I/O failure no longer masks an existing backend exception; a log
   failure on otherwise-successful sampling still surfaces.
5. Multi-GPU is detected from prepared clone options or registered additional
   models before upstream creates its thread pool.

Final: minor (deferred): object_info checker may raise AttributeError when a
linked source node is a malformed non-dict, instead of returning an error list.
This did not affect the shipped valid API examples or runtime node. Fixed in
the edit continuation; it is no longer an open issue.

Final: Ruling (historical): edit kernels, real-model/GPU execution, quality and
performance were outside this diagnostic review. The edit continuation below
supersedes the implementation scope; real-model qualification remains open.

No inference, weights, external publication, merge or remote push was performed.

Final verification after fixes: **95 passed** with the optional source-contract
tests enabled; `compileall` and `git diff --check` passed; all three API examples
passed graph-only validation. No dedicated linter/type checker or live
`/object_info` validation was run. Work remains in the requested directory on
`diagnostic/krea2-afm` at that stage; current work is on `feat/krea2-afm-edit`.

## Edit continuation

Base: b81d70f. New local branch: feat/krea2-afm-edit.

- Reused the existing plan's interfaces and invariant tests; no source forward
  copy, global patch, model download, dependency install, or inference.
- Independent executor lanes implemented numeric kernels and offline tooling;
  the leader integrated node/runtime/diagnostics and owns final verification.
- Numeric kernels retain full image-grid FFT, all-key softmax, explicit scale,
  and per-head output restoration. Local autocast is disabled for FP32 paths.
- Editing retains one baseline, returns untouched text/unselected sample rows,
  and discards all partial replacements on failure. Zero original adds no hooks.
- Qualified profiles remain empty. Auto matches actual memory-adjusted chunk
  sizes before using any future profile, including dtype/precision/environment.
- The previously deferred malformed-node checker issue is fixed with a regression test.
- GPU benchmark ordering is tested with mocks only. CPU runs never report CUDA peaks.
- Edit UI/API examples passed static linkage/widget checks; real UI loading is unverified.
- Dry-run matrix preparation saved seven prompts; all seven remained `not_submitted`.

Before final review: full source-contract-enabled suite **225 passed**;
one GPU test deselected. Additional review regressions are recorded below.

## Resumed implementation verification (2026-09-30)

- Matrix runner now generates one UUID `invocation_id` per invocation and injects
  `krea2_afm/<run-id>-<invocation-id>.jsonl` after all node-input overrides.
  Manifest and mapped server logs use the same identity. Missing current logs
  stay `unverified`; previous fixed-name logs are never used as evidence.
- Multiple matrix conditions carry their own mapped log path through execution.
  A two-condition regression proves each summary is attached to its own record.
- Profile qualification canonicalizes dtype aliases and rejects mixed activation
  dtype reports. Required strengths cannot be combined across FP16/BF16.
  Existing candidate-precision and numerical-threshold guards remain in force.
- Regression evidence: runner identity tests failed before the fix; the extra
  multi-run case also failed before its fix. Runner suite: **11 passed**.
  Mixed-dtype regression failed before the fix; report suite: **15 passed**.
- Final full suite with `KREA2_COMFY_SOURCE_ROOT=.reference/upstream-9d80841`:
  **241 passed, 1 GPU test deselected**. No source-contract tests skipped.
- `compileall`, `git diff --check`, and edit API graph validation passed.
  Dry-run manifest assertions confirmed seven `not_submitted` conditions,
  seven `unverified` metrics, unique log paths, and matching log inputs in all
  six AFM prompts (the seventh is true baseline).
- Dedicated lint/typecheck tools (`ruff`, `mypy`, `pyright`) remain unavailable;
  none were installed. No real-model, GPU, server/UI load or quality validation.

### Remaining minor diagnostic issues

- Unsupported branch/mask reasons can be counted twice in one attention call.
- `profile_not_qualified` is in attention metadata but absent from aggregate
  `RunState.reasons`. Edit and fallback counts are unaffected.

These counters remain deferred from the prior kernel/runtime review. Profiles
remain empty; no GPU support, speed, VRAM or image-quality claim is made.

Independent final review of the resumed tools/tests and logging/selection
boundaries found **no unresolved issues**. Its own targeted suite passed all
26 tests. Previously reviewed kernel/runtime internals were not reopened beyond
those integrations. Real-model/GPU/performance/quality/profile adoption remain
outside this review. Work is retained locally on `feat/krea2-afm-edit`.

## 2026-10-15 refactoring implementation — UNVERIFIED

The user explicitly deferred verification for this implementation turn. T0–T5 source/test/tooling changes are recorded in `refactoring-implementation-2026-10-15.md`. No tests, compilation, lint, typecheck, workflow dry-run, server/UI, model/GPU, performance validation or independent code review was executed for this diff. Prior 241-pass evidence applies to pre-refactor HEAD 944e507 only. T6–T8 and all validation gates remain pending.
