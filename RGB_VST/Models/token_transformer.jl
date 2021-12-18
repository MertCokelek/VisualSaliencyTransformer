# includeonly("transformer_block.jl", "mutable struct Block")
include("transformer_block.jl")
include("utils.jl")
export Attention, Token_transformer

mutable struct TTAttention
    dim; num_heads; in_dim; qkv_bias; qk_scale; attn_drop; proj_drop;
    
    scale; qkv; proj;
    function TTAttention(;dim=784, in_dim=0, num_heads=8, qkv_bias=false, qk_scale=false, attn_drop=0., proj_drop=0.)
        self = new(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_heads = num_heads
        self.in_dim = in_dim
        
        head_dim = dim ÷ num_heads
        
        self.scale = qk_scale || head_dim^-0.5
        
        # Seperating qkv.
        println("TTAttention ", dim, " ", in_dim)
        self.qkv = Linear(dim, in_dim*3, bias=qkv_bias)
        
        # self.q = Linear(dim, in_dim, bias=qkv_bias)
        # self.k = Linear(dim, in_dim, bias=qkv_bias)
        # self.v = Linear(dim, in_dim, bias=qkv_bias)
        
        self.attn_drop = Dropout(attn_drop)

        self.proj = Linear(in_dim, in_dim)
        self.proj_drop = Dropout(proj_drop)
        return self
    end
end


function(self::TTAttention)(x)
    B, N, C = size(x)
    
    qkv = permutedims(reshape(self.qkv(x), (B, N, 3, self.num_heads, self.in_dim)), (3,1,4,2,5)) # Old (VST PyTorch implementation) Merged QKV. 
    q, k, v = qkv[1], qkv[2], qkv[3]
    # q, k, v = qkv[1,:,:,:,:], qkv[2,:,:,:,:], qkv[3,:,:,:,:]

    # q = permutedims(reshape(self.q(x), (B, N, 1, self.num_heads, self.in_dim)), (3,1,4,2,5))[1,:,:,:,:]
    # k = permutedims(reshape(self.k(x), (B, N, 1, self.num_heads, self.in_dim)), (3,1,4,2,5))[1,:,:,:,:]
    # v = permutedims(reshape(self.v(x), (B, N, 1, self.num_heads, self.in_dim)), (3,1,4,2,5))[1,:,:,:,:]
    
    attn = (q * transposedims(k, -2, -1)) .* self.scale
    softmax_dim = length(size(attn))
    attn = softmax(attn, dims=softmax_dim)
    attn = self.attn_drop(attn)
    
    x = reshape(transposedims(attn * v, 1, 2), (B, N, self.in_dim))
    x = self.proj(x)
    x = self.proj_drop(x, self.proj_drop)
    return x
end 

using Knet.Ops21:gelu
mutable struct Token_transformer
    dim; in_dim; num_heads; mlp_ratio; qkv_bias; qk_scale; drop; attn_drop; drop_path; act_layer; norm_layer;
    
    norm1; attn; norm2; mlp;
    
    function Token_transformer(dim, in_dim, num_heads; mlp_ratio=1., qkv_bias=false, qk_scale=false, drop=0., attn_drop=0., drop_path=0., proj_drop=0., act_layer="gelu", norm_layer="LayerNorm")
        self = new(dim, in_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)
        println("Token trnsformer ", dim, " ", in_dim)
        self.norm1=LayerNorm(dim)
        self.attn = TTAttention(;dim=dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop, proj_drop=proj_drop)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(in_dim)
        self.mlp = Mlp(in_dim;hidden_features=convert(Int32, in_dim*mlp_ratio), out_features=in_dim, act_layer=gelu, drop=drop)
        return self
    end
end


function (self::Token_transformer)(x)
    x = self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    # x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
end