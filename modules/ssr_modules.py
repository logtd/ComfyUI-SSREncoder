import torch
import torch.nn.functional as F
from torch import nn

import comfy.ops
ops = comfy.ops.disable_weight_init
from comfy.ldm.modules.attention import optimized_attention


class SSRAligner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        query_dim = 768
        inner_dim = 512
        cross_dim = 1024
        self.heads = 8

        self.to_q = ops.Linear(query_dim, inner_dim, bias=False)
        self.to_k = ops.Linear(cross_dim, inner_dim, bias=False)
        
        self.to_v = nn.ModuleList([])
        self.to_out = nn.ModuleList([])
        for _ in range(6):
            self.to_v.append(ops.Linear(cross_dim, inner_dim, bias=False))
            self.to_out.append(ops.Linear(inner_dim, query_dim, bias=False))
        self.norm = ops.LayerNorm(query_dim)

    def forward(self, x, context_list):
        q = self.to_q(x)
        k = self.to_k(context_list[-1])

        outs = []
        for i in range(6):
            v = self.to_v[i](context_list[i])
            attn_out = optimized_attention(q, k, v, self.heads)
            outs.append(self.to_out[i](attn_out))

        return self.norm(torch.cat(outs, dim=1)) # [1, 77*6, 768]
    

# me being lazy
ssr_caa_dims = {
    'input': [320, 320, 640, 640, 1280, 1280],
    'middle': [1280],
    'output': [1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
}


class SSRCrossAttention(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.heads = 8

        self.to_k = ops.Linear(768, hidden_size, bias=False)
        self.to_v = ops.Linear(768, hidden_size, bias=False)

        self.pos_contexts = [] # [{ 'scale': int, 'embed': tensor }, ...]
        self.neg_contexts = []

    def to(self, device):
        super().to(device)
        for i in range(len(self.pos_contexts)):
            self.pos_contexts[i]['embed'] = self.pos_contexts[i]['embed'].to(device)

        for i in range(len(self.neg_contexts)):
            self.neg_contexts[i]['embed'] = self.neg_contexts[i]['embed'].to(device)

        return self

    def forward(self, q, k, v, extra_options):

        hidden_states = optimized_attention(q, k, v, extra_options['n_heads'])
        cond_or_uncond = extra_options['cond_or_uncond']
        len_conds = len(cond_or_uncond)
        batch_size = len(q)//len_conds

        for cond_idx, cond in enumerate(cond_or_uncond):
            contexts = self.pos_contexts if cond == 0 else self.neg_contexts
            q_cond = q[cond_idx*batch_size:(cond_idx+1)*batch_size]
            for context in contexts:
                scale = context['scale']
                embed = context['embed']
                ssr_k = self.to_k(embed).repeat(batch_size, 1, 1)
                ssr_v = self.to_v(embed).repeat(batch_size, 1, 1)
                ssr_hidden_states = optimized_attention(q_cond, ssr_k, ssr_v, self.heads)

                hidden_states[cond_idx*batch_size:(cond_idx+1)*batch_size] += scale * ssr_hidden_states

        return hidden_states
