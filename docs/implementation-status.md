# Implementation status

Plan: `Krea2_AFM_Codex_Implementation_Plan_2026-09-30.md`, revision 2.

## Current milestone

Implement Tasks 0–1 (diagnostics) in the requested directory, then hand off
for the intermediate real-model GPU check required by the plan. Tasks 2–5
remain pending that evidence. No edit kernel or qualified GPU profile is
claimed by this milestone.

## Decisions and preflight

- Task 0 → Task 1: immutable `AFMConfig`, native Krea2 capabilities and patch
  conflict detection feed runtime installation; interfaces agree.
- Task 1 → Tasks 2–4: layout and progress types retain the plan's interfaces.
- Ruling: work directly in the newly initialized repository on
  `diagnostic/krea2-afm`; there was no existing checkout to isolate. The
  requested output directory remains the installable package.
- Ruling: use the supplied design/plan and its explicit diagnostic milestone
  under the user's autonomous execution instruction. Further design approval
  is unnecessary; GPU evidence is still required before the edit milestone.
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
- [ ] Tasks 2–5: FFT and edit kernels, numerical/GPU/quality/performance tests.

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

Task 1 exposes no edit settings or pretend implementation. The intermediate
GPU gate and Tasks 2–5 are still open, as required by the supplied plan.

## Final review

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
This does not affect the shipped valid API examples or runtime node.

Final: Ruling: edit kernels, real-model/GPU execution, quality and performance
were outside this diagnostic review. They remain explicit open milestones,
not implied support. Cost: no AFM edit or GPU capability can be claimed yet.

No inference, weights, external publication, merge or remote push was performed.

Final verification after fixes: **95 passed** with the optional source-contract
tests enabled; `compileall` and `git diff --check` passed; all three API examples
passed graph-only validation. No dedicated linter/type checker or live
`/object_info` validation was run. Work remains in the requested directory on
`diagnostic/krea2-afm`; there was no existing base branch or remote to integrate.
