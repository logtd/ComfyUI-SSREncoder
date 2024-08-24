from .nodes.apply_ssr_attention_node import ApplySSRNode
from .nodes.encode_ssr_query_node import EncodeSSRQueryNode
from .nodes.load_ssr_aligner_node import LoadSSRAlignerNode
from .nodes.load_ssr_attention_node import LoadSSRAttentionNode


NODE_CLASS_MAPPINGS = {
    'ApplySSR': ApplySSRNode,
    'EncodeSSRQuery': EncodeSSRQueryNode,
    'LoadSSRAligner': LoadSSRAlignerNode,
    'LoadSSRAttention': LoadSSRAttentionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ApplySSR': 'SSR Apply Model',
    'EncodeSSRQuery': 'SSR Encode Query',
    'LoadSSRAligner': 'SSR Load Aligner',
    'LoadSSRAttention': 'SSR Load Attentions',
}
