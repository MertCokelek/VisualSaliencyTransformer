include("t2t_vit.jl")
using Knet:KnetArray;

x = KnetArray(rand(4, 3, 224, 224));

# tokens_to_token = T2T_module();

# tokens_to_token(x)


rgb_backbone = T2T_ViT(tokens_type="transformer", 
                        embed_dim=384,
                        depth=14,
                        num_heads=6,
                        mlp_ratio=3.
                        );

rgb_backbone(x)