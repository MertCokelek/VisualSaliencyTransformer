export T2T_ViT

include("token_transformer.jl")
include("utils.jl")
using PyCall

# @pyimport torch
@pyimport torch.nn as nn
mutable struct T2T_module
    """
    Tokens-to-Token encoding module
    """
    img_size; tokens_type;
    in_chans
    embed_dim; token_dim;
    
    
    soft_split0; soft_split1; soft_split2
    attention1; attention2
    project
    num_patches
    
    function T2T_module(img_size=224, 
                        tokens_type="transformer", 
                        in_chans=3, 
                        embed_dim=768, 
                        token_dim=64)
        self = new(img_size, tokens_type, in_chans, embed_dim, token_dim)
        
        if tokens_type == "transformer"
            println("adopt transformer encoder for tokens-to-token")
            # unfold operation will be done in functionally.
            self.soft_split0 = Unfold(7, 4, 2)
            self.soft_split1 = Unfold(3, 2, 1)
            self.soft_split2 = Unfold(3, 2, 1)            
            # self.attention1 = Token_performer(in_chans*7*7, token_dim; kernel_ratio=0.5)
            # self.attention2 = Token_performer(token_dim*3*3, token_dim; kernel_ratio=0.5)

            self.attention1 = Token_transformer(in_chans * 7 * 7, token_dim, 1; mlp_ratio=1.0)
            self.attention2 = Token_transformer(token_dim * 3 * 3, token_dim, 1; mlp_ratio=1.0)
            self.project = Linear(token_dim * 3 * 3, embed_dim)
        end
        
        self.num_patches = (img_size ÷ (4 * 2 * 2)) * (img_size ÷ (4 * 2 * 2))  # there are 3 soft split, stride are 4,2,2 seperately

        return self
        """
        TODO / NOTE: initially, let's just consider 'performer' model. 'transformer' & 'convolution' can be done later.
        Performer and Transformer implemented. Pretrained weights are supplied for transformer, so it will be used. 
        TODO: Unfold currently taken from PyTorch. Might need conversion between datatypes.
        """
    end
    
end

function (self::T2T_module)(x)
    # step0: soft split
    x = permutedims(self.soft_split0(x), [1, 3, 2])
    # x = permutedims(self.soft_split0(x), [2, 1, 3])
    println("T2T_module forward: ", size(x), " ", typeof(x));

    
    # x [B, 56*56, 147=7*7*3]
    # iteration1: restructurization/reconstruction
    x_1_4 = self.attention1(x)
    B, new_HW, C = size(x_1_4)
    x = reshape(permutedims(x_1_4, [2, 3]), (B, C, convert(Int, sqrt(new_HW)), convert(Int, sqrt(new_HW))))
    # iteration1: soft_split
    x = permutedims(self.soft_split1(x), [2, 3])

    # iteration2: restructurization/reconstruction
    x_1_8 = self.attention2(x)
    B, new_HW, C = size(x_1_4)
    x = reshape(permutedims(x_1_8, [2, 3]), (B, C, convert(Int, sqrt(new_HW)), convert(Int, sqrt(new_HW))))
    # iteration1: soft_split
    x = permutedims(self.soft_split2(x), [2, 3])

    # final_tokens
    x = self.project(x)
    
    return x, x_1_8, x_1_4
end
    


mutable struct T2T_ViT; 
    img_size; tokens_type
    in_chans; num_classes; num_features
    embed_dim; depth
    num_heads; mlp_ratio
    qkv_bias; qk_scale
    drop_rate; attn_drop_rate; drop_path_rate; 
    norm_layer;

    blocks
    norm
    head
    tokens_to_token
    cls_token
    pos_embed

    function T2T_ViT(;img_size=224, 
            tokens_type="transformer", 
            in_chans=3, num_classes=1000,
            embed_dim=784,depth=12,
            num_heads=12, mlp_ratio=4., 
            qkv_bias=false, qk_scale=false, 
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
            norm_layer="LayerNorm"
            )
            
        self = new(img_size, tokens_type, 
                    in_chans, num_classes, 
                    embed_dim,depth,num_heads,
                    mlp_ratio,qkv_bias,qk_scale,
                    drop_rate,attn_drop_rate,
                    drop_path_rate,norm_layer)
        self.num_features = embed_dim

        #self.tokens_to_token = T2T_module()
        self.tokens_to_token = T2T_module(
                img_size,tokens_type,in_chans, embed_dim)
        self.cls_token = convert(KnetArray, zeros(1,1,embed_dim))
        num_patches = self.tokens_to_token.num_patches
        self.pos_embed = reshape(get_sinusoid_encoding(num_patches + 1, embed_dim).w, (1, num_patches+1, embed_dim))  #
        

        # NOTE: In PyTorch implementation, dropout is added as a portable layer.
        # We will add it as an operation, in the forward pass. 
        # UPDATE: probably we won't need it, since we'll use pretrained weights. 
        
        self.drop_rate = drop_rate
        dpr = [i for i in 0:0.25/depth:0.25] # Stochastic depth decay rule
        self.blocks = [Block(dim=embed_dim, 
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale,
                            drop=drop_rate, 
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[i], 
                            act_layer=gelu,
                            norm_layer=norm_layer) for i in 1:depth]
        @assert norm_layer == "LayerNorm"                   
        self.norm = LayerNorm(embed_dim)
        # Classifier head
        self.head = Linear(embed_dim, num_classes)
#         For initializing weights, probably we won't need the rest.
#         We will use pretrained weights.
        return self
    end
end


function expand(x, B)
    """
    Helper function for torch.Tensor.expand()    expand(x, B) is equivalent to x.expand(B, -1, -1)
    """
    res = x
    dimx = length(size(x))
    for i in 1:B-1
        res = cat(res, x, dims=dimx+1)
    end
    permute_order = [i for i in 1:dimx]
    insert!(permute_order, 1, dimx+1)
    res = permutedims(res, permute_order)
end

function (self::T2T_ViT)(x)
    B = size(x)[1]
    x, x_1_8, x_1_4 = self.tokens_to_token(x)
    
    cls_tokens = expand(self.cls_token, B) # self.cls_token.expand(B, -1, -1) in Torch
    x = hcat(cls_tokens, x)
    x += self.pos_embed
    # x = self.pos_drop # Dropout will not be used for now.
    
    # T2T-ViT backbone
    for blk in self.blocks
        x = blk(x)
    end
    
    x = self.norm(x)
    return x[:,2:end,:], x_1_8, x_1_4
end
