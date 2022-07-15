include("utils.jl")
include("token_transformer.jl")
include("unfold.jl")

mutable struct T2T_module
    """
    Tokens-to-Token encoding module
    """
    img_size; 
    tokens_type;
    in_chans;
    embed_dim;
    token_dim;
    
    soft_split0; soft_split1; soft_split2;
    attention1; attention2;
    project;
    num_patches;
    
    function T2T_module(img_size=224, tokens_type="transformer", in_chans=3, embed_dim=768, token_dim=64)
        self = new(img_size, tokens_type, in_chans, embed_dim, token_dim)
        
        if tokens_type == "transformer"
            println("adopt transformer encoder for tokens-to-token")
            # unfold operation will be done in functionally.
            self.soft_split0 = Unfold4D(7, 4, 2)
            self.soft_split1 = Unfold4D(3, 2, 1)
            self.soft_split2 = Unfold4D(3, 2, 1)            

            self.attention1 = Token_transformer(in_chans * 7 * 7, token_dim, 1; mlp_ratio=1.0)
            self.attention2 = Token_transformer(token_dim * 3 * 3, token_dim, 1; mlp_ratio=1.0)
            self.project = Linear(token_dim * 3 * 3, embed_dim);
            return self
        end         
    end
    
    
    function (self::T2T_module)(x)
        println("Testing T2T Module Forward pass")
        # step0: soft split
        x = self.soft_split0(x)
        x_1_4 = self.attention1(x)               # bs, 3136, 64 
        
        B, new_HW, C = size(x_1_4)
        x = reshape(permutedims(x_1_4, (1, 3, 2)),  (B, C, int(sqrt(new_HW)), int(sqrt(new_HW))))
        x = self.soft_split1(x)
        x_1_8 = self.attention2(x)               # bs, 784, 64
        
        B, new_HW, C = size(x_1_8)
        x = reshape(permutedims(x_1_8, (1, 3, 2)),  (B, C, int(sqrt(new_HW)), int(sqrt(new_HW))))
        x = self.soft_split2(x)
        x = self.project(x)                      # bs, 196, 384
        println("End of T2T module");@show size(x);
        return x
    end
        
end
