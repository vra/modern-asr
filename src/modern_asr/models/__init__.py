"""Modern ASR model adapters.

Each submodule implements one or more `ASRModel` subclasses and registers them
via `@register_model`. Importing this package triggers auto-discovery.
"""

from modern_asr.core.registry import auto_discover_models

# Trigger registration of all model adapters
auto_discover_models("modern_asr.models")
