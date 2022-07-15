include("t2t_vit.jl")
include("load_pretrained_weights.jl")
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


model_path = "/home/mcokelek21/Desktop/Github/VST/RGB_VST/pretrained_model/80.7_T2T_ViT_t_14.pth.tar"

# compare(model_path, rgb_backbone);
rgb_backbone = loadPretrainedWeightsT2TViT(model_path, rgb_backbone); 

rgb_backbone(x)