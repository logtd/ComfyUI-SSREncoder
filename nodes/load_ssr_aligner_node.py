import os

from safetensors.torch import load_file

import folder_paths
from folder_paths import models_dir

from ..modules.ssr_modules import SSRAligner


SSR_PATH = os.path.join(models_dir, 'ssr')
os.makedirs(SSR_PATH, exist_ok=True)


class LoadSSRAlignerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "checkpoint": (os.listdir(SSR_PATH), )}}

    RETURN_TYPES = ("SSR_ALIGNER",)
    FUNCTION = "load"

    CATEGORY = "ssr"

    def load(self, checkpoint):
        checkpoint_path = os.path.join(SSR_PATH, checkpoint)
        state_dict = load_file(checkpoint_path)
        ssr_aligner = SSRAligner()
        ssr_aligner.load_state_dict(state_dict)
        return (ssr_aligner,)