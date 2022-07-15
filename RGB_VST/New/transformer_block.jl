mutable struct Linear; 
    w; b;
    function Linear(input::Int,outputs...; bias=true)
        self = new(param(outputs...,input), bias ? param0(outputs...) : nothing)
        if bias; self.b = reshape(self.b, 1, 1, :);end
        return self
    end
    
    function (l::Linear)(x)
        W1,W2,X1,X2 = size(l.w)[1:end-1], size(l.w)[end], size(x,3), size(x)[1:2]; 
        y = l.w * reshape(x,X1,:)
        y = reshape(y, X2..., W1...)
        if l.b !== nothing; y = y .+ l.b; end
        return y
    end
end


mutable struct Mlp
    in_features; hidden_features; out_features; act_layer; drop_rate;
    
    drop; fc1; act; fc2;
    
    function Mlp(in_features; hidden_features=nothing, out_features=nothing, act_layer::Function=gelu, drop_rate=0.)
        self = new(in_features, hidden_features, out_features, act_layer, drop_rate)
        
        out_features = out_features === nothing ? in_features : out_features
        hidden_features = hidden_features === nothing ? in_features : hidden_features
        
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer#() ????
        self.fc2 = Linear(hidden_features, out_features)
        # self.drop = drop # called in forward pass
        self.drop = Dropout(drop_rate)
        return self
    end
    
    function (self::Mlp)(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x
    end
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
        # self.attn = Attention(;dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop)
        self.attn = Attention(;dim=dim, in_dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop, proj_drop=drop)
    
        # TODO: DropPath: If drop_path != 0., Adapt https://github.com/rwightman/pytorch-image-models/blob/f7d210d759beb00a3d0834a3ce2d93f6e17f3d38/timm/models/layers/drop.py#L160 to Julia
        # self.drop_path = drop_path > 0.? DropPath(drop_path) : nn.Identity()  # ????
                
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = convert(Int, dim * mlp_ratio)
        self.mlp = Mlp(dim;   hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop_rate=drop)
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