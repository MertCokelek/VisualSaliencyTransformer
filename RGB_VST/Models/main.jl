include("t2t_vit.jl")
include("load_pretrained_weights.jl")
include("dataset.jl")

# Start model definition.
rgb_backbone = T2T_ViT(tokens_type="transformer", 
                        embed_dim=384,
                        depth=14,
                        num_heads=6,
                        mlp_ratio=3.
                        );

# model_path = "/home/mcokelek21/Downloads/80.7_T2T_ViT_t_14.pth.tar"
model_path = "/home/mcokelek21/Desktop/Github/VST/RGB_VST/pretrained_model/80.7_T2T_ViT_t_14.pth.tar"

compare(model_path, rgb_backbone);
model_pt = loadPretrainedWeightsT2TViT(model_path, rgb_backbone); 
# End model definition.


# Start dataset definition
dataset_train = load_data(train_image_paths, train_mask_paths, train_contour_paths);

x_train, y_train = dataset_train;

size(x_train), size(y_train)

batch_size = 4
train_data = Knet.minibatch(x_train, y_train, batch_size)

x, y = first(train_data);
imgs = x[:,:,1,:]
msks = y[:,:,1,:]
cnts = y[:,:,2,:]
size(imgs), size(msks), size(cnts)

x_batch = zeros(Float32, (batch_size, 3, 224, 224))
y_mask_batch = zeros(Float32, (batch_size, 224, 224))
y_cont_batch = zeros(Float32, (batch_size, 224, 224))
for i in 1:batch_size
    println("Sample ", i, " preprocessing.")
    sample, mask, cont = preprocess_img.(eachslice(cat(imgs, msks, cnts, dims=4), dims=3))[i];   
#     mask = channelview(Gray.(mask)) # Convert gt to grayscale
#     cont = channelview(Gray.(cont)) # Convert gt to grayscale
    sample = convert(Array{Float32},channelview(sample));
    ymask = convert(Array{Float32}, channelview(Gray.(mask)));
    ycont = convert(Array{Float32}, channelview(Gray.(cont)));
    
    x_batch[i,:,:,:] = sample;
    y_mask_batch[i,:,:] = ymask;
    y_cont_batch[i,:,:] = ycont;
end

# End dataset definition