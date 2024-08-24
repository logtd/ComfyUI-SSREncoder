import comfy.model_management

from ..modules.clip_vision_override import encode_image


class EncodeSSRQueryNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_scale": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01}),
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The positive image query"}), 
                "negative_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The negative image query"}), 
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "clip_vision": ("CLIP_VISION",),
                "ssr_aligner": ("SSR_ALIGNER",),
                "image": ("IMAGE",),
            },
            "optional": {
                "prev_ssr_embeds": ("SSR_EMBEDS",)
            }
        }
    RETURN_TYPES = ("SSR_EMBEDS",)
    FUNCTION = "encode"
    CATEGORY = "ssr"

    def _encode_text(self, text, clip):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return cond

    def encode(self, 
               positive_scale,
               positive, 
               negative_scale, 
               negative, 
               clip,
               clip_vision, 
               ssr_aligner, 
               image, 
               prev_ssr_embeds=None):
        if prev_ssr_embeds is None:
            prev_ssr_embeds = { 'negative': [], 'positive': [] }
        else:
            prev_ssr_embeds = { 
                'negative': [*prev_ssr_embeds['negative']], 
                'positive': [*prev_ssr_embeds['positive']] 
            }


        intermediate_device = comfy.model_management.intermediate_device()
        torch_device = comfy.model_management.get_torch_device()
        image_embeds = encode_image(clip_vision, image)[3::4]

        ssr_aligner.to(torch_device)
        pos_embed = None
        if positive and positive_scale > 0:
            pos_embed = self._encode_text(positive, clip).to(torch_device)
            pos_embed = ssr_aligner(pos_embed, image_embeds).to(intermediate_device)
            prev_ssr_embeds['positive'].append({ 'scale': positive_scale, 'embed': pos_embed })
        if negative and negative_scale > 0:
            neg_embed = self._encode_text(negative, clip).to(torch_device)
            neg_embed = ssr_aligner(neg_embed, image_embeds).to(intermediate_device)
            prev_ssr_embeds['negative'].append({ 'scale': negative_scale, 'embed': neg_embed })
        
        ssr_aligner.to(intermediate_device)

        return (prev_ssr_embeds, )