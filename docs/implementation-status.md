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
- License review: do not copy unlicensed Anima source. New implementation
  follows the supplied specification; third-party provenance is documented.

## Validation environment

- Windows; Python 3.10.11; PyTorch 2.10.0+cpu; pytest 9.0.2.
- No local ComfyUI import, CUDA runtime, or Krea2 inference environment.
- ComfyUI authoring skill bundle completeness: 37 checks passed.

## Tasks

- [ ] Task 0: configuration, capabilities, conflict checks, tests, provenance.
- [ ] Task 1: diagnostic runtime, backend dispatch, nodes, examples, tests.
- [ ] Intermediate GPU gate: true baseline/discover/observe on Turbo and
  rectangle; confirm backend, grid, txtfusion exclusion, CFG, run isolation.
- [ ] Tasks 2–5: FFT and edit kernels, numerical/GPU/quality/performance tests.

Test evidence and final review findings are appended during implementation.
