import os

from safetensors.torch import load_file

from folder_paths import models_dir

from ..modules.ssr_modules import SSRCrossAttention, ssr_caa_dims


SSR_PATH = os.path.join(models_dir, 'ssr')
os.makedirs(SSR_PATH, exist_ok=True)


class LoadSSRAttentionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "checkpoint": (os.listdir(SSR_PATH), )}}

    RETURN_TYPES = ("SSR_ATTENTIONS",)
    FUNCTION = "load"

    CATEGORY = "ssr"

    def load(self, checkpoint):
        checkpoint_path = os.path.join(SSR_PATH, checkpoint)
        state_dict = load_file(checkpoint_path)
        ssr_idx = 1
        ssr_attentions = {}
        for idx, i in enumerate([1,2,3,4,5,6]):
            ssr_aligner = SSRCrossAttention(ssr_caa_dims['input'][idx])
            attn_dict = {
                'to_k.weight': state_dict[f'{ssr_idx}.to_k_SSR.weight'],
                'to_v.weight': state_dict[f'{ssr_idx}.to_v_SSR.weight'],
            }
            ssr_aligner.load_state_dict(attn_dict)
            ssr_attentions[('input', i)] = ssr_aligner
            ssr_idx += 2

        for idx, i in enumerate([3,4,5,6,7,8,9,10,11]):
            ssr_aligner = SSRCrossAttention(ssr_caa_dims['output'][idx])
            attn_dict = {
                'to_k.weight': state_dict[f'{ssr_idx}.to_k_SSR.weight'],
                'to_v.weight': state_dict[f'{ssr_idx}.to_v_SSR.weight'],
            }
            ssr_aligner.load_state_dict(attn_dict)
            ssr_attentions[('output', i)] = ssr_aligner
            ssr_idx += 2
            
        ssr_aligner = SSRCrossAttention(ssr_caa_dims['middle'][0])
        attn_dict = {
            'to_k.weight': state_dict[f'{ssr_idx}.to_k_SSR.weight'],
            'to_v.weight': state_dict[f'{ssr_idx}.to_v_SSR.weight'],
        }
        ssr_aligner.load_state_dict(attn_dict)
        ssr_attentions[('middle', 0)] = ssr_aligner

        return (ssr_attentions,)