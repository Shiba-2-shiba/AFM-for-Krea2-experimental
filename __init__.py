"""ComfyUI V3 extension entry point; host imports are deferred until loading."""


async def comfy_entrypoint():
    from .nodes import Krea2AFMExtension
    return Krea2AFMExtension()


__all__ = ["comfy_entrypoint"]
