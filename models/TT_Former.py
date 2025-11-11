"""
Time Text Former
"""
import sys
sys.path.append('/dataYYF/dataWX/SJ/Time-QA/')
import math
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import Mlp, DropPath
from timm.layers.helpers import to_2tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils.position_coding import LearnablePositionalEmbedding, SinusoidalPositionalEncoding,RotaryPositionalEncoding
from models.layers.attention import InstructTimeAttention

from models.layers.attention import GatedCrossAttention
class ITformer(nn.Module):
    """
    ITformer - MoPE (Mixture of Prompt Experts) 版本实现
    """
    def __init__(self, args):
        super(ITformer, self).__init__()
        
        self.tt_d_model = args.tt_d_model
        self.prefix_num = args.prefix_num
        self.num_experts = 3 # K=3 (text, ts, fusion)

        # 步骤 1. 定义“模态提示专家”库 (K=3)
        # 形状: (K, n, d)
        self.expert_lib = nn.Parameter(torch.randn(
            self.num_experts, 
            self.prefix_num, 
            self.tt_d_model
        ))
        torch.nn.init.normal_(self.expert_lib, std=.02)

        # 步骤 2. 定义基于查询的“模态”路由
        # 输入是 AvgPool(H_q)，形状为 (B, d)
        self.router = nn.Sequential(
            nn.Linear(self.tt_d_model, self.tt_d_model // 2), # FFN_route
            nn.ReLU(),
            nn.Linear(self.tt_d_model // 2, self.num_experts), # 输出 K=3 个权重
            nn.Softmax(dim=-1)
        )

        # 步骤 4. 定义单阶段“门控交叉注意力 (GCA)”
        self.gca_fusion = GatedCrossAttention(
            dim=self.tt_d_model,
            num_heads=args.tt_n_heads,
            proj_drop=args.tt_dropout,
            norm_layer=nn.LayerNorm
        )
        
        # 保留原有的位置编码模块 (用于 memory)
        self.time_pos = SinusoidalPositionalEncoding(args.tt_d_model)
        self.var_pos = LearnablePositionalEmbedding(args.tt_d_model)
        self.cycle_pos = RotaryPositionalEncoding(args.tt_d_model)

    def apply_memory_pos_enc(self, memory, stage):
        """
        保留原有的对 memory (时序数据) 应用位置编码的逻辑
        注意：原逻辑中 stage 是一个 tensor，这里假设它是一个 list 或 1D tensor。
        """
        # (这部分逻辑从原 ITformer.forward 复制和调整)
        
        # 确保 stage 是在 CPU 上的 list，以便索引
        if isinstance(stage, torch.Tensor):
            stage = stage.cpu().tolist()

        # 按 stage 分离 memory
        cycle_indices = [i for i, s in enumerate(stage) if s not in [3, 4]]
        cross_cycle_indices = [i for i, s in enumerate(stage) if s in [3, 4]]
        
        # 创建一个空的 memory_with_pos 来按顺序填充
        memory_with_pos = torch.zeros_like(memory)

        # 1. 处理 cycle_memory (stage 1, 2)
        if cycle_indices:
            cycle_memory = memory[cycle_indices]
            b, l, v, d = cycle_memory.shape
            
            # 应用 time_pos
            cycle_memory_time = cycle_memory.view(b * l, v, d)
            cycle_memory_time = cycle_memory_time + self.time_pos(cycle_memory_time)
            cycle_memory = cycle_memory_time.view(b, l, v, d)
            
            # 应用 var_pos
            cycle_memory_var = cycle_memory.permute(0, 2, 1, 3).reshape(b * v, l, d) # (b*v, l, d)
            cycle_memory_var = cycle_memory_var + self.var_pos(cycle_memory_var)
            cycle_memory = cycle_memory_var.view(b, v, l, d).permute(0, 2, 1, 3) # (b, l, v, d)
            
            memory_with_pos[cycle_indices] = cycle_memory

        # 2. 处理 cross_cycle_memory (stage 3, 4)
        if cross_cycle_indices:
            cross_cycle_memory = memory[cross_cycle_indices]
            b, l, v, d = cross_cycle_memory.shape
            
            # 应用 cycle_pos
            cross_cycle_memory_cycle = cross_cycle_memory.view(b * v, l, d)
            cross_cycle_memory_cycle = cross_cycle_memory_cycle + self.cycle_pos(cross_cycle_memory_cycle)
            cross_cycle_memory = cross_cycle_memory_cycle.view(b, v, l, d).permute(0, 2, 1, 3) # (b, l, v, d)
            
            # 应用 var_pos
            cross_cycle_memory_var = cross_cycle_memory.permute(0, 2, 1, 3).reshape(b * v, l, d)
            cross_cycle_memory_var = cross_cycle_memory_var + self.var_pos(cross_cycle_memory_var)
            cross_cycle_memory = cross_cycle_memory_var.view(b, v, l, d).permute(0, 2, 1, 3) # (b, l, v, d)
            
            memory_with_pos[cross_cycle_indices] = cross_cycle_memory
            
        return memory_with_pos


    def forward(self, x, memory, stage=None, attn_mask=None):
        """
        x (torch.Tensor): H_q (查询嵌入), 形状 (B, L_q, d)
        memory (torch.Tensor): T (时序嵌入), 形状 (B, V, L_ts, d)
        stage (torch.Tensor): 批次中每个样本的 stage
        """
        
        # 步骤 2. 基于查询的“模态”路由
        # router_input = AvgPool(H_q), 形状 (B, d)
        router_input = torch.mean(x, dim=1) 
        # routing_weights, 形状 (B, K)
        routing_weights = self.router(router_input) 

        # 步骤 3. 动态构建“融合查询” I*
        # self.expert_lib 形状 (K, n, d)
        # routing_weights 形状 (B, K)
        # 使用 einsum 实现加权平均: (B, K) * (K, n, d) -> (B, n, d)
        dynamic_instruct_query_I_star = torch.einsum(
            'bk,knd->bnd', 
            routing_weights, 
            self.expert_lib
        )

        # 4. GCA 融合之前，对 memory (T) 应用位置编码
        memory_with_pos = self.apply_memory_pos_enc(memory, stage)
        
        # 步骤 4. 单阶段门控交叉注意力（GCA）融合
        # H_fusion = GCA(Q=I*, K=T, V=T)
        H_fusion = self.gca_fusion(
            query=dynamic_instruct_query_I_star,
            memory=memory_with_pos
        )
        
        # H_fusion (B, n, d) 即为最终的融合输出 (tt_embeds)
        return H_fusion
# class SeqCrossAttention(nn.Module):
#     def __init__(
#             self,
#             dim,
#             num_heads=8,
#             qkv_bias=False,
#             qk_norm=False,
#             attn_drop=0.,
#             proj_drop=0.,
#             norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, query, key_value, attn_mask=None):
#         B, N, C = query.shape
#         _, V, L, _ = key_value.shape

#         # Reshape Key and Value to focus only on L (time) dimension
#         key_value = key_value.view(B * V, L, C)

#         # Compute Query, Key, and Value projections
#         q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         kv = self.kv_proj(key_value).reshape(B * V, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         k, v = kv.unbind(0)

#         # Adjust batch size for Key and Value
#         k = k.view(B, V, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, V * L, self.head_dim)
#         v = v.view(B, V, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, V * L, self.head_dim)

#         # Apply normalization (if any)
#         q = self.q_norm(q).reshape(B * self.num_heads, N, self.head_dim)
#         k = self.k_norm(k)

#         # Scaled Dot-Product Attention over L dimension
#         x = F.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=attn_mask,
#             dropout_p=self.attn_drop.p if self.training else 0.
#         )

#         # Reshape and project output
#         x = x.view(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
    
# class SeqAttBlock(nn.Module):

#     def __init__(
#             self,
#             dim,
#             num_heads,
#             qkv_bias=False,
#             qk_norm=False,
#             proj_drop=0.,
#             attn_drop=0.,
#             init_values=None,
#             drop_path=0.,
#             norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn_seq = SeqCrossAttention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,
#         )

#         self.drop_path1 = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()
#         self.proj = nn.Linear(dim, dim)

#     def forward(self, x, key_value, attn_mask):
#         x_input = x
#         x = self.norm1(x)
#         key_value = self.norm1(key_value)

#         # key_value = torch.reshape(
#         #     key_value, (-1, key_value.shape[-2], key_value.shape[-1]))
#         x = self.attn_seq(x, key_value, attn_mask)
#         x = x_input + self.drop_path1(x)
#         return x

# class VarCrossAttention(nn.Module):

#     def __init__(
#             self,
#             dim,
#             num_heads=8,
#             qkv_bias=False,
#             qk_norm=False,
#             attn_drop=0.,
#             proj_drop=0.,
#             norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, query, key_value, attn_mask=None):
#         B, N, C = query.shape
#         _, V, L, _ = key_value.shape

#         # Reshape Key and Value to focus only on V (variable) dimension
#         key_value = key_value.view(B * L, V, C)

#         # Compute Query, Key, and Value projections
#         q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         kv = self.kv_proj(key_value).reshape(B * L, V, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         k, v = kv.unbind(0)

#         # Adjust batch size for Key and Value
#         k = k.view(B, L, self.num_heads, V, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, L * V, self.head_dim)
#         v = v.view(B, L, self.num_heads, V, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, L * V, self.head_dim)

#         # Apply normalization (if any)
#         q = self.q_norm(q).reshape(B * self.num_heads, N, self.head_dim)
#         k = self.k_norm(k)

#         # Scaled Dot-Product Attention over V dimension
#         x = F.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=attn_mask,
#             dropout_p=self.attn_drop.p if self.training else 0.
#         )

#         # Reshape and project output
#         x = x.view(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class VarAttBlock(nn.Module):

#     def __init__(
#             self,
#             dim,
#             num_heads,
#             qkv_bias=False,
#             qk_norm=False,
#             proj_drop=0.,
#             attn_drop=0.,
#             init_values=None,
#             drop_path=0.,
#             norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn_var = VarCrossAttention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,
#         )

#         self.drop_path1 = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()
#         self.proj = nn.Linear(dim, dim)

#     def forward(self, x, key_value, attn_mask):
#         x_input = x
#         x = self.norm1(x)
#         key_value = self.norm1(key_value)

#         x = self.attn_var(x, key_value, attn_mask)
#         x = x_input + self.drop_path1(x)
#         return x

# class SeqAttention(nn.Module):

#     def __init__(
#             self,
#             dim,
#             num_heads=8,
#             qkv_bias=False,
#             qk_norm=False,
#             attn_drop=0.,
#             proj_drop=0.,
#             norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, attn_mask=None):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
#                                   self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)
#         x = F.scaled_dot_product_attention(
#             q, k, v,  attn_mask=attn_mask,
#             dropout_p=self.attn_drop.p if self.training else 0.,
#         )

#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class SelfAttBlock(nn.Module):

#     def __init__(
#             self,
#             dim,
#             num_heads,
#             qkv_bias=False,
#             qk_norm=False,
#             proj_drop=0.,
#             attn_drop=0.,
#             init_values=None,
#             drop_path=0.,
#             norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn_seq = SeqAttention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,
#         )

#         self.drop_path1 = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x, attn_mask=None):
#         x_input = x
#         x = self.norm1(x)
#         x = self.attn_seq(x, attn_mask)
#         x = x_input + self.drop_path1(x)
#         return x


# class ITAttBlock(nn.Module):

#     def __init__(
#             self,
#             dim,
#             num_heads,
#             qkv_bias=False,
#             qk_norm=False,
#             proj_drop=0.,
#             attn_drop=0.,
#             init_values=None,
#             drop_path=0.,
#             norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn_it = InstructTimeAttention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,

#         )
#         self.drop_path1 = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x,memory, attn_mask=None):
#         x_input = x
#         x = self.attn_it(x, memory,attn_mask)
#         x = x_input + self.norm1(self.drop_path1(x))
#         return x

# class DecoderBasicBlock(nn.Module):

#     def __init__(
#             self,
#             dim,
#             num_heads,
#             mlp_ratio=4.0,
#             qkv_bias=False,
#             qk_norm=False,
#             proj_drop=0.,
#             attn_drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             prefix_num=10,
#     ):
#         super().__init__()

#         self.prefix_num = prefix_num

#         self.self_attn = SelfAttBlock(
#               dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
#             attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, norm_layer=norm_layer
#         ) 

#         self.it_attn = ITAttBlock(
#             dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
#             attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, norm_layer=norm_layer
#         )

        
#         self.feed_forward_prefix = nn.Sequential(
#             norm_layer(dim),
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             act_layer(),
#             nn.Dropout(proj_drop),
#             nn.Linear(int(dim * mlp_ratio), dim),
#         )

#         self.feed_forward_instruct = nn.Sequential(
#             norm_layer(dim),
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             act_layer(),
#             nn.Dropout(proj_drop),
#             nn.Linear(int(dim * mlp_ratio), dim),
#         )


#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x, memory, attn_mask=None):

#         # Self-attention block
#         x = x + self.self_attn(x, attn_mask)
#         prefix  =x[:, :self.prefix_num, :]
#         x = self.feed_forward_instruct(x)+x
#         # Cross-attention block 10token vs b,n,c,d中的n
#         prefix = prefix + self.it_attn(prefix, memory, attn_mask)
#         # 10token vs b,n,c,d中的c
#         # Feed forward block
#         prefix = prefix + self.feed_forward_prefix(prefix)
#         # Concatenate prefix and x
#         x = torch.cat([prefix, x[:, self.prefix_num:, :]], dim=1)
#         return x


# class ITformer(nn.Module):
#     def __init__(self, args):
#         super(ITformer, self).__init__()
#         self.layers = nn.ModuleList([
#             DecoderBasicBlock(
#                 dim=args.tt_d_model,
#                 num_heads=args.tt_n_heads,
#                 mlp_ratio=4.,
#                 qkv_bias=True,
#                 qk_norm=False,
#                 proj_drop=args.tt_dropout,
#                 attn_drop=args.tt_dropout,
#                 drop_path=0.,
#                 act_layer=nn.GELU,
#                 norm_layer=nn.LayerNorm,
#                 prefix_num=args.prefix_num
#             ) for _ in range(args.tt_layers)
#         ])
#         self.norm = nn.LayerNorm(args.tt_d_model)


#         #time posi
#         self.time_pos = SinusoidalPositionalEncoding(args.tt_d_model)
#         #variable posi
#         self.var_pos = LearnablePositionalEmbedding(args.tt_d_model)
#         #instruction posi
#         self.instruc_pos = SinusoidalPositionalEncoding(args.tt_d_model)
#         # cycle posi
#         self.cycle_pos = RotaryPositionalEncoding(args.tt_d_model)

#         #prefix num
#         self.prefix_num = args.prefix_num
#         self.prefix_token = nn.Parameter(torch.randn(1, args.prefix_num, args.tt_d_model))
#     def forward(self, x, memory, stage=None,attn_mask=None):

#         # Add prefix token to x 
#         x = torch.cat([self.prefix_token.repeat(x.shape[0], 1, 1), x], dim=1)
#         # Positional encoding
#         # Apply positional encoding to x
#         x = x + self.instruc_pos(x)

#         #Stage是list,找出stage中等于3,4的位置
#         cycle_index = [i for i in stage if i != 3 and i != 4]
#         cross_cycle_index = [i for i in stage if i == 3 or i == 4]

#         cycle_memory = memory[cycle_index, :, :, :]
#         cross_cycle_memory = memory[cross_cycle_index, :, :, :]

#         # Reshape and apply positional encoding to memory at time dimension
#         b, l, v, d = cycle_memory.shape
#         cycle_memory = cycle_memory.view(b * l, v, d)
#         cycle_memory = cycle_memory + self.time_pos(cycle_memory)
#         cycle_memory = cycle_memory.view(b, l, v, d)


#         # Reshape and apply positional encoding to memory at cycle dimension
#         b, l, v, d = cross_cycle_memory.shape
#         cross_cycle_memory = cross_cycle_memory.view(b * v, l, d)
#         cross_cycle_memory = cross_cycle_memory + self.cycle_pos(cross_cycle_memory)
#         cross_cycle_memory = cross_cycle_memory.view(b, l, v, d)

#         memory = torch.cat([cycle_memory, cross_cycle_memory], dim=0)

#         # Reshape and apply positional encoding to memory at var dimension
#         b, v, l, d = memory.shape
#         memory = memory.view(b * l, v, d)
#         memory = memory + self.var_pos(memory)
#         memory = memory.view(b, l, v, d)


#         for layer in self.layers:
#             x = layer(x, memory, attn_mask)
#         x = self.norm(x)


#         return x[:, :self.prefix_num, :]

# def count_parameters(model):
#     """统计模型中可训练参数的总数"""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)



# if __name__ == "__main__":

    # dim = 64
    # num_heads = 8
    # seq_len = 20
    # var_num = 5
    # memory_len = 30
    # batch_size = 2

    # x = torch.randn(batch_size, seq_len, dim)
    # memory = torch.randn(batch_size, var_num,memory_len, dim)
    # attn_mask = None

    # decoder_block = DecoderBasicBlock(
    #     dim=dim, num_heads=num_heads, qkv_bias=True, proj_drop=0.1, attn_drop=0.1
    # )
    # output = decoder_block(x, memory, attn_mask)
    # print("DecoderBasicBlock Output Shape:", output.shape)
    # class Args:
    #     def __init__(self):
    #         self.tt_d_model = 64
    #         self.tt_n_heads = 8
    #         self.tt_layers = 6
    #         self.tt_dropout = 0.1
    #         self.prefix_num = 10

    # args = Args()
    # model = TTformer(args)

    # x = torch.randn(batch_size, seq_len, dim)
    # memory = torch.randn(batch_size, var_num, memory_len, dim)
    # attn_mask = None
    # stage = [1,2]
    # output = model(x, memory,stage, attn_mask)
    # print("Model Output Shape:", output.shape)
    # class Args:
    #     def __init__(self):
    #         self.tt_d_model = 512
    #         self.tt_n_heads = 8
    #         self.tt_layers = 4
    #         self.tt_dropout = 0.1
    #         self.prefix_num = 10

    # args = Args()
    # model = ITformer(args)

    # # 打印可训练参数量
    # total_trainable_params = count_parameters(model)
    # print(f"Total Trainable Parameters: {total_trainable_params:,}")

    # # 可选：打印每一层的参数量
    # print("\nLayer-wise Parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.numel():,}")