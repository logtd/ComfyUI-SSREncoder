
class ApplySSRNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ssr_attentions": ("SSR_ATTENTIONS",), 
                "ssr_embeds": ("SSR_EMBEDS",),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "ssr"

    def apply(self, model, ssr_attentions, ssr_embeds):
        model = model.clone()

        for key, ssr_attention in ssr_attentions.items():
            ssr_attention.pos_contexts = ssr_embeds['positive']
            ssr_attention.neg_contexts = ssr_embeds['negative']
            block_name, block_number = key
            model.set_model_attn2_replace(ssr_attention, block_name, block_number)

        return (model,)

