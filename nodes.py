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
            description=("Experimental Krea2 AFM diagnostics. Discover/observe preserve original "
                         "attention. Editing is not implemented in this diagnostic release."),
            search_aliases=["krea2 afm", "attention frequency modulation"],
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("mode", options=["off", "discover", "observe"], default="observe"),
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
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, mode="observe", block_scope="0-3", branch_mode="positive_only",
                start_percent=0.0, end_percent=1.0, fail_mode="raise", debug_level="summary",
                jsonl_path="krea2_afm/diagnostics.jsonl") -> io.NodeOutput:
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
                           fail_mode=fail_mode, debug_level=debug_level, jsonl_path=jsonl_path)
        return io.NodeOutput(install_krea2_afm(model, config))


class Krea2AFMExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Krea2AFMModelPatch]
