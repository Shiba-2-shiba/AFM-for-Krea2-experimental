# Sources and licensing

The diagnostic implementation is newly written from the supplied design and
the public ComfyUI API contracts. It does not import or copy the Anima AFM
implementation or the Krea2T Enhancer implementation at runtime.

- `AFM-for-anima-experimental`, reference revision
  `596abc87f4db983aaceea19b437119367f45eb23`: conceptual reference only.
  No LICENSE file was present in the supplied local snapshot; source code
  has not been copied. A later FFT port must resolve licensing before copying.
- `ComfyUI-Krea2T-Enhancer`, reference revision
  `e1f60f79e168c62b1cb1cf773dfb3ae4ee2292e2`: MIT, copyright (c) 2026
  capitan01R. Consulted for Krea2 shape and clone patterns; no code copied.
- ComfyUI revision `9d80841aa1990305cc7a280c5ea505f317efcc2d`:
  public model, attention, sampler and wrapper API reference. No model forward
  or global monkey patch is included in this package. Downloaded source used
  for local contract checks is excluded from the repository under `.reference/`.

The supplied implementation plan retains its own provenance. The LICENSE
below applies to the newly written implementation, tests and documentation.
