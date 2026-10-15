from pathlib import Path

from comfy_api.latest import ComfyExtension, io

from .afm.config import AFMConfig
from .afm.runtime import install_krea2_afm


class Krea2AFMModelPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Krea2AFMModelPatch",
            display_name="Krea2 AFM Model Patch",
            category="model_patches/krea2",
            description=("Experimental Krea2 image-to-text attention frequency modulation. "
                         "Discover/observe preserve original attention. Auto uses manual until GPU qualified."),
            search_aliases=["krea2 afm", "attention frequency modulation"],
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("mode", options=["off", "discover", "observe", "edit"], default="observe"),
                io.String.Input("block_scope", default="0-3",
                                tooltip="Body block indices: 0-3, 0,2 or all. Discover logs all blocks."),
                io.Combo.Input("branch_mode", options=["positive_only", "negative_only", "both"],
                               default="positive_only"),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("fail_mode", options=["raise", "fallback_original"], default="raise"),
                io.Combo.Input("debug_level", options=["off", "summary", "verbose"], default="summary"),
                io.String.Input("jsonl_path", default="krea2_afm/diagnostics.jsonl",
                                tooltip="A .jsonl path under output/krea2_afm/. Empty disables file logging."),
                io.Float.Input("strength", default=0.05, min=-1.0, max=2.0, step=0.01),
                io.Combo.Input("implementation", options=["auto", "manual", "augmented"], default="auto"),
                io.Float.Input("cutoff", default=0.25, min=0.01, max=0.99, step=0.01),
                io.Combo.Input("schedule", options=["curve", "frequency_neutral"],
                               default="curve", advanced=True,
                               tooltip="curve shifts LF to HF gain. frequency_neutral uses "
                                       "LF=HF=1+strength/2 with the same DC/normalization settings; "
                                       "it does not match total edit magnitude and is not global "
                                       "attention-temperature scaling."),
                io.Boolean.Input("preserve_dc", default=True, advanced=True),
                io.Boolean.Input("soft_mask", default=True, advanced=True),
                io.Float.Input("mask_width", default=0.05, min=0.001, max=1.0, step=0.01, advanced=True),
                io.Combo.Input("zero_strength_mode", options=["original", "manual", "augmented"],
                               default="original", advanced=True),
                io.Int.Input("head_chunk_size", default=2, min=1, max=64, advanced=True),
                io.Int.Input("query_chunk_size", default=128, min=1, max=4096, advanced=True),
                io.Float.Input("max_workspace_mib", default=512.0, min=1.0, max=65536.0, step=16.0, advanced=True),
                io.Combo.Input("augmented_precision", options=["activation", "float32"],
                               default="activation", advanced=True),
                io.Combo.Input("spectral_diag", options=["off", "sampled"], default="off", advanced=True),
                io.String.Input("diagnostic_block_scope", default="0", advanced=True),
                io.Int.Input("diagnostic_every_n_steps", default=1, min=1, max=1000, advanced=True),
                io.Int.Input("diagnostic_head_limit", default=1, min=1, max=8, advanced=True),
                io.Int.Input("diagnostic_query_chunk_size", default=128, min=1, max=512, advanced=True),
                io.Int.Input("diagnostic_top_k", default=4, min=1, max=64, advanced=True),
                io.Boolean.Input("compare_kernels", default=False, advanced=True,
                                 tooltip="Bounded same-QKV manual/augmented comparisons; adds significant work."),
                io.Boolean.Input("benchmark", default=False, advanced=True,
                                 tooltip="Synchronize CUDA and measure server-side peak memory and elapsed time."),
                io.Combo.Input("normalization_mode", options=["joint", "preserve_text_mass"],
                               default="joint", optional=True, advanced=True,
                               tooltip="Preserve total text attention mass with the manual kernel."),
                io.String.Input("diagnostic_head_indices", default="", optional=True, advanced=True,
                                tooltip="Physical head indices, e.g. 0,16,32,47. Empty uses head limit."),
                io.Int.Input("diagnostic_radial_bins", default=32, min=1, max=256,
                             optional=True, advanced=True),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, mode="observe", block_scope="0-3", branch_mode="positive_only",
                start_percent=0.0, end_percent=1.0, fail_mode="raise", debug_level="summary",
                jsonl_path="krea2_afm/diagnostics.jsonl", strength=0.05, implementation="auto",
                cutoff=0.25, schedule="curve", preserve_dc=True, soft_mask=True, mask_width=0.05,
                zero_strength_mode="original", head_chunk_size=2, query_chunk_size=128,
                max_workspace_mib=512.0, augmented_precision="activation", spectral_diag="off",
                diagnostic_block_scope="0", diagnostic_every_n_steps=1, diagnostic_head_limit=1,
                diagnostic_query_chunk_size=128, diagnostic_top_k=4, compare_kernels=False,
                benchmark=False, normalization_mode="joint", diagnostic_head_indices="",
                diagnostic_radial_bins=32) -> io.NodeOutput:
        if jsonl_path:
            import folder_paths
            relative = Path(jsonl_path)
            output_root = Path(folder_paths.get_output_directory()).resolve()
            path = (output_root / relative).resolve()
            if (relative.is_absolute() or ".." in relative.parts or relative.suffix != ".jsonl"
                    or not path.is_relative_to(output_root / "krea2_afm")):
                raise ValueError("jsonl_path must be a relative .jsonl path under krea2_afm/")
            jsonl_path = str(path)
        config = AFMConfig(mode=mode, block_scope=block_scope, branch_mode=branch_mode,
                           start_percent=start_percent, end_percent=end_percent,
                           fail_mode=fail_mode, debug_level=debug_level, jsonl_path=jsonl_path,
                           strength=strength, implementation=implementation, cutoff=cutoff,
                           schedule=schedule, preserve_dc=preserve_dc, soft_mask=soft_mask,
                           mask_width=mask_width, zero_strength_mode=zero_strength_mode,
                           head_chunk_size=head_chunk_size, query_chunk_size=query_chunk_size,
                           max_workspace_mib=max_workspace_mib, augmented_precision=augmented_precision,
                           spectral_diag=spectral_diag, diagnostic_block_scope=diagnostic_block_scope,
                           diagnostic_every_n_steps=diagnostic_every_n_steps,
                           diagnostic_head_limit=diagnostic_head_limit,
                           diagnostic_query_chunk_size=diagnostic_query_chunk_size,
                           diagnostic_top_k=diagnostic_top_k, compare_kernels=compare_kernels,
                           benchmark=benchmark, normalization_mode=normalization_mode,
                           diagnostic_head_indices=diagnostic_head_indices,
                           diagnostic_radial_bins=diagnostic_radial_bins)
        return io.NodeOutput(install_krea2_afm(model, config))


class Krea2AFMExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Krea2AFMModelPatch]
