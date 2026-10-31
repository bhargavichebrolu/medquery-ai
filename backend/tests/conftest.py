"""Test-session guards.

Hugging Face offline mode is forced before any test imports huggingface_hub, so no
test can download model weights. Tests that need real models load them from the
local cache only and are opt-in (marker `models`, MEDQUERY_RUN_MODEL_TESTS=1).
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
