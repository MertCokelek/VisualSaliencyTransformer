using Knet: atype
using Knet.Ops21:gelu
include("utils.jl")
export get_sinusoid_encoding, Mlp, Block



struct get_sinusoid_encoding 
    w 
end

function get_sinusoid_encoding(n_position, d_hid; λ=10000, atype=atype())
    x = exp.((0:2:d_hid-1) .* -(log(λ)/d_hid)) * (0:n_position-1)'
    pe = zeros(d_hid, n_position)
    pe[1:2:end,:] = sin.(x)
    pe[2:2:end,:] = cos.(x)
    get_sinusoid_encoding(atype(pe))
end

function (l::get_sinusoid_encoding)(x)
    x .+ l.w[:,1:size(x,2)]
end


mutable struct Mlp
    in_features; hidden_features; out_features; act_layer; drop;
    
    fc1; act; fc2;
    function Mlp(in_features; hidden_features=nothing, out_features=nothing, act_layer::Function=gelu, drop=0.)
        self = new(in_features, hidden_features, out_features, act_layer, drop)
        
        out_features = out_features === nothing ? in_features : out_features
        hidden_features = hidden_features === nothing ? in_features : hidden_features
        
        self.fc1 = Linear(in_features, hidden_features)
        self.act = gelu#() ????
        self.fc2 = Linear(hidden_features, out_features)
        # self.drop = drop # called in forward pass
        self.drop = Dropout(drop)
        return self
    end
    
    function (self::Mlp)(x)
        x = self.drop(self.act(self.fc1(x)), drop)
        x = self.drop(self.fc2(x), drop)
        return x
    end
end

function transposedims(x, dim1, dim2)
    """
    Helper function equivalent to torch.Tensor.transpose(dim1, dim2)
    
    """
    size_ = [i for i in size(x)]
    dims = length(size_)
    dim1_ = dim1 > 0 ? dim1 : dims+dim1+1
    dim2_ = dim2 > 0 ? dim2 : dims+dim2+1
    
    size_[dim1_] = dim2_
    size_[dim2_] = dim1_
    
    return permutedims(x, size_)
end

mutable struct TransformerAttention
    dim; num_heads; qkv_bias; qk_scale; attn_drop; proj_drop;
    
    scale; qkv; proj;
    function TransformerAttention(;dim=784, num_heads=8, qkv_bias=false, qk_scale=false, attn_drop=0., proj_drop=0.)
        self = new(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_heads = num_heads
        head_dim = dim ÷ num_heads
        
        self.scale = qk_scale || head_dim^-0.5
        
        self.qkv = Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        return self
    end
end
function(self::TransformerAttention)(x)
    B, N, C = size(x)
    qkv = permutedims(reshape(self.qkv(x), (B, N, 3, self.num_heads, C ÷ self.num_heads)), (3,1,4,2,5))
    q, k, v = qkv[1], qkv[1], qkv[2]
    
    attn = (q * transposedims(k, -2, -1)) .* self.scale
    softmax_dim = length(size(attn))
    attn = softmax(attn, dims=softmax_dim)
    attn = self.attn_drop(attn)
    
    x = reshape(transposedims(attn * v, 1, 2), (B, N, C))
    x = self.proj(x)
    x = self.proj_drop(x, self.proj_drop)
    return x
end 


mutable struct Block
    dim; num_heads; mlp_ratio; qkv_bias; qk_scale; drop; attn_drop;
    drop_path; act_layer; norm_layer;
    
    norm1;
    attn;
    norm2;
    mlp;
    
    function Block(;dim=784, num_heads=8, mlp_ratio=4.0, qkv_bias=false, qk_scale=false, drop=0., attn_drop=0.,
        drop_path=0., act_layer::Function=gelu, norm_layer="LayerNorm")
    
        self = new(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)
        
        @assert norm_layer == "LayerNorm"
        self.norm1 = LayerNorm(dim)
        self.attn = TransformerAttention(;dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop)
        # self.attn =  Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # TODO: DropPath: If drop_path != 0., Adapt https://github.com/rwightman/pytorch-image-models/blob/f7d210d759beb00a3d0834a3ce2d93f6e17f3d38/timm/models/layers/drop.py#L160 to Julia
        # self.drop_path = drop_path > 0.? DropPath(drop_path) : nn.Identity()  # ????
                
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = convert(Int, dim * mlp_ratio)
        self.mlp = Mlp(dim; hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        return self
    end
end

function (self::Block)(x)
    # x = x + self.drop_path(self.attn(self.norm1(x)))
    # x = x + self.drop_path(self.mlp(self.norm2(x)))
    
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x
end 
