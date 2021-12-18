include("t2t_vit.jl")
include("load_pretrained_weights.jl")
rgb_backbone = T2T_ViT(tokens_type="transformer", 
                        embed_dim=384,
                        depth=14,
                        num_heads=6,
                        mlp_ratio=3.
                        );

model_path = "/home/mertcokelek/Downloads/80.7_T2T_ViT_t_14.pth.tar"
model_pt = loadPretrainedWeightsT2TViT(model_path, rgb_backbone); # compare(model_path, rgb_backbone);

println("Hello World")