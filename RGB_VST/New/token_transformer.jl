using Knet.Ops21:gelu
using Knet:bmm, softmax
include("transformer_block.jl")

mutable struct Attention
    dim; 
    num_heads; 
    in_dim; 
    qkv_bias; 
    qk_scale; 
    attn_drop; 
    proj_drop;
    
    scale; 
    qkv; 
    proj;

    
    mutable struct QKV; w; b; end

    function QKV(input::Int,outputs...; bias=true)
        QKV(param(outputs...,input), bias ? param0(outputs...) : nothing)
    end

    function (l::QKV)(x)
        W1,W2,X1,X2 = size(l.w)[1:end-1], size(l.w)[end], size(x,3), size(x)[1:2]; 
        @assert W2===X1
        y = reshape(l.w,:,W2) * reshape(x,X1,:)
        y = reshape(y, W1..., X2...)
        if l.b !== nothing; y = y .+ l.b; end
        return permutedims(y, (2, 3, 1))
    end
    
    mutable struct Proj; w; b; end

    function Proj(input::Int,outputs...; bias=true)
        Proj(param(outputs...,input),
               bias ? param0(outputs...) : nothing)
    end

    function (l::Proj)(x)
        W1,W2,X1,X2 = size(l.w)[1:end-1], size(l.w)[end], size(x,1), size(x)[2:end];
        @assert W2===X1
        y = reshape(l.w,:,W2) * reshape(x,X1,:)
        y = reshape(y, W1..., X2...)
        if l.b !== nothing; y = y .+ l.b; end
        return y
    end
       
    
    function Attention(;dim=784, in_dim=0, num_heads=8, qkv_bias=false, qk_scale=false, attn_drop=0., proj_drop=0.)
        self = new(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_heads = num_heads
        self.in_dim = in_dim
        
        head_dim = dim ÷ num_heads
        
        self.scale = qk_scale || head_dim^-0.5
        
        # Seperating qkv.
        # self.qkv = QKV(dim, in_dim*3, bias=qkv_bias)
        self.qkv = Linear(dim, in_dim*3, bias=qkv_bias)
        
        self.attn_drop = Dropout(attn_drop)

        # self.proj = Proj(in_dim, in_dim)
        self.proj = Linear(in_dim, in_dim)
        self.proj_drop = Dropout(proj_drop)
        return self
    end    
    
    function (self::Attention)(x)
        # size(x): (B, 3136, 147) (bs, 56*56, 7*7*3)
        qkv = self.qkv(x)
        bs, hw, c = size(qkv)
        qkv = reshape(qkv, (bs, 1, hw, 3, c ÷ 3))
        q, k, v = qkv[:,:,:,1,:], qkv[:,:,:,2,:], qkv[:,:,:,3,:]
        
        q = permutedims(q, (3, 4, 2, 1))
        v = permutedims(v, (3, 4, 2, 1)) 
        k = permutedims(k, (4, 3, 2, 1))
        
        attn = bmm(k,q)                             # @size s (T1,T2,H,B)
        attn = attn * eltype(attn)(self.scale)      # @size s (T1,T2,H,B)
        attn = softmax(attn, dims=1);               # @size s (T1,T2,H,B)
        
        x = bmm(v,attn)                             # @size c (V2,T2,H,B)
        x = permutedims(x, (1,3,2,4))               # @size c (V2,H,T2,B)
        x = reshape(x, :, size(x,3), size(x,4))     # @size c (O1,T2,B)
        x = permutedims(x, (3, 1, 2))
        x = self.proj(x)                            # @size o (O2,T2,B)
        
        result = permutedims(v[:,:,1,:], (3, 1, 2)) + x
        return result
    end 
end


mutable struct Token_transformer
    dim; in_dim; num_heads; mlp_ratio; qkv_bias; qk_scale; drop; attn_drop; drop_path; act_layer; norm_layer;
    
    norm1; attn; norm2; mlp;
    
    function Token_transformer(dim, in_dim, num_heads; mlp_ratio=1, qkv_bias=false, qk_scale=false, drop=0., attn_drop=0., drop_path=0., proj_drop=0., act_layer=gelu, norm_layer="LayerNorm")
        self = new(dim, in_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.norm1=LayerNorm(dim)
        self.attn = Attention(;dim=dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop, proj_drop=proj_drop)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(in_dim)
        self.mlp = Mlp(in_dim;hidden_features=convert(Int64, in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop_rate=drop)
        # self.mlp = Mlp(in_dim;hidden_features=in_dim*mlp_ratio, out_features=in_dim, act_layer=gelu, drop_rate=drop)
        return self
    end
end


function (self::Token_transformer)(x)
    x = self.norm1(x)
    x = self.attn(x)
    x = x + self.mlp(self.norm2(x))
    # x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
end
