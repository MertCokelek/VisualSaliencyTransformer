export compare, loadPretrainedWeightsT2TViT
using PyCall
@pyimport torch

# model_path = "/home/mertcokelek/Downloads/80.7_T2T_ViT_t_14.pth.tar"
# model_path = "/home/mcokelek21/Downloads/80.7_T2T_ViT_t_14.pth.tar"

function print_pretrained(path)
    weights = torch.load(path, map_location=torch.device("cpu"))["state_dict_ema"];
    weights = sort(collect(weights), by = x->x[1])

    for (k, v) in weights
        @show k, v.shape
    end
end

# fun

function compare(path, model)
    weights = torch.load(path, map_location=torch.device("cpu"))["state_dict_ema"];
    for i in 1:14
        # blocks
        @assert size(model.blocks[i].attn.qkv.w) === size(Param(atype(weights["blocks.$(i-1).attn.qkv.weight"][:cpu]()[:numpy]()))); 
        
        @assert size(model.blocks[i].attn.proj.w) === size(Param(atype(weights["blocks.$(i-1).attn.proj.weight"][:cpu]()[:numpy]()))); 
        @assert size(model.blocks[i].attn.proj.b) === size(Param(atype(weights["blocks.$(i-1).attn.proj.bias"][:cpu]()[:numpy]()))); 
        
        @assert size(model.blocks[i].mlp.fc1.w) === size(Param(atype(weights["blocks.$(i-1).mlp.fc1.weight"][:cpu]()[:numpy]()))); 
        @assert size(model.blocks[i].mlp.fc1.b) === size(Param(atype(weights["blocks.$(i-1).mlp.fc1.bias"][:cpu]()[:numpy]()))); 
        
        
        @assert size(model.blocks[i].mlp.fc2.w) === size(Param(atype(weights["blocks.$(i-1).mlp.fc2.weight"][:cpu]()[:numpy]()))); 
        @assert size(model.blocks[i].mlp.fc2.b) === size(Param(atype(weights["blocks.$(i-1).mlp.fc2.bias"][:cpu]()[:numpy]()))); 
        
        @assert size(model.blocks[i].norm1.a) === size(Param(atype(weights["blocks.$(i-1).norm1.weight"][:cpu]()[:numpy]()))); 
        @assert size(model.blocks[i].norm1.b) === size(Param(atype(weights["blocks.$(i-1).norm1.bias"][:cpu]()[:numpy]()))); 
        
        @assert size(model.blocks[i].norm2.a) === size(Param(atype(weights["blocks.$(i-1).norm2.weight"][:cpu]()[:numpy]()))); 
        @assert size(model.blocks[i].norm2.b) === size(Param(atype(weights["blocks.$(i-1).norm2.bias"][:cpu]()[:numpy]()))); 
   end

    # tokens_to_token
          # attn 1
    @assert size(model.tokens_to_token.attention1.attn.proj.w) === size(Param(atype(weights["tokens_to_token.attention1.attn.proj.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention1.attn.proj.b) === size(Param(atype(weights["tokens_to_token.attention1.attn.proj.bias"][:cpu]()[:numpy]()))); 

    @assert size(model.tokens_to_token.attention1.attn.qkv.w) === size(Param(atype(weights["tokens_to_token.attention1.attn.qkv.weight"][:cpu]()[:numpy]()))); 

    @assert size(model.tokens_to_token.attention1.mlp.fc1.w) === size(Param(atype(weights["tokens_to_token.attention1.mlp.fc1.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention1.mlp.fc1.b) === size(Param(atype(weights["tokens_to_token.attention1.mlp.fc1.bias"][:cpu]()[:numpy]()))); 
   
    @assert size(model.tokens_to_token.attention1.mlp.fc2.w) === size(Param(atype(weights["tokens_to_token.attention1.mlp.fc2.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention1.mlp.fc2.b) === size(Param(atype(weights["tokens_to_token.attention1.mlp.fc2.bias"][:cpu]()[:numpy]()))); 

    @assert size(model.tokens_to_token.attention1.norm1.a) === size(Param(atype(weights["tokens_to_token.attention1.norm1.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention1.norm1.b) === size(Param(atype(weights["tokens_to_token.attention1.norm1.bias"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention1.norm2.a) === size(Param(atype(weights["tokens_to_token.attention1.norm2.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention1.norm2.b) === size(Param(atype(weights["tokens_to_token.attention1.norm2.bias"][:cpu]()[:numpy]()))); 
    
          # attn2
    @assert size(model.tokens_to_token.attention2.attn.proj.w) === size(Param(atype(weights["tokens_to_token.attention2.attn.proj.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention2.attn.proj.b) === size(Param(atype(weights["tokens_to_token.attention2.attn.proj.bias"][:cpu]()[:numpy]()))); 

    @assert size(model.tokens_to_token.attention2.attn.qkv.w) === size(Param(atype(weights["tokens_to_token.attention2.attn.qkv.weight"][:cpu]()[:numpy]()))); 

    @assert size(model.tokens_to_token.attention2.mlp.fc1.w) === size(Param(atype(weights["tokens_to_token.attention2.mlp.fc1.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention2.mlp.fc1.b) === size(Param(atype(weights["tokens_to_token.attention2.mlp.fc1.bias"][:cpu]()[:numpy]()))); 
   
    @assert size(model.tokens_to_token.attention2.mlp.fc2.w) === size(Param(atype(weights["tokens_to_token.attention2.mlp.fc2.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention2.mlp.fc2.b) === size(Param(atype(weights["tokens_to_token.attention2.mlp.fc2.bias"][:cpu]()[:numpy]()))); 

    @assert size(model.tokens_to_token.attention2.norm1.a) === size(Param(atype(weights["tokens_to_token.attention2.norm1.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention2.norm1.b) === size(Param(atype(weights["tokens_to_token.attention2.norm1.bias"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention2.norm2.a) === size(Param(atype(weights["tokens_to_token.attention2.norm2.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.attention2.norm2.b) === size(Param(atype(weights["tokens_to_token.attention2.norm2.bias"][:cpu]()[:numpy]()))); 

         # project
    @assert size(model.tokens_to_token.project.w) === size(Param(atype(weights["tokens_to_token.project.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.tokens_to_token.project.b) === size(Param(atype(weights["tokens_to_token.project.bias"][:cpu]()[:numpy]()))); 
    
    
    @assert size(model.cls_token) === size(Param(atype(weights["cls_token"][:cpu]()[:numpy]()))); 

    @assert size(model.head.w) === size(Param(atype(weights["head.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.head.b) === size(Param(atype(weights["head.bias"][:cpu]()[:numpy]()))); 

    @assert size(model.norm.a) === size(Param(atype(weights["norm.weight"][:cpu]()[:numpy]()))); 
    @assert size(model.norm.b) === size(Param(atype(weights["norm.bias"][:cpu]()[:numpy]()))); 

    @assert size(model.pos_embed) === size(Param(atype(weights["pos_embed"][:cpu]()[:numpy]()))); 

    println("Model structure is correct!")
end
 

function loadPretrainedWeightsT2TViT(path, model)
    weights = torch.load(path, map_location=torch.device("cpu"))["state_dict_ema"];
    for i in 1:14
        # blocks
        model.blocks[i].attn.qkv.w = Param(atype(weights["blocks.$(i-1).attn.qkv.weight"][:cpu]()[:numpy]()))
        
        model.blocks[i].attn.proj.w = Param(atype(weights["blocks.$(i-1).attn.proj.weight"][:cpu]()[:numpy]()))
        model.blocks[i].attn.proj.b = Param(atype(weights["blocks.$(i-1).attn.proj.bias"][:cpu]()[:numpy]()))
        
        model.blocks[i].mlp.fc1.w = Param(atype(weights["blocks.$(i-1).mlp.fc1.weight"][:cpu]()[:numpy]()))
        model.blocks[i].mlp.fc1.b = Param(atype(weights["blocks.$(i-1).mlp.fc1.bias"][:cpu]()[:numpy]()))
        
        
        model.blocks[i].mlp.fc2.w = Param(atype(weights["blocks.$(i-1).mlp.fc2.weight"][:cpu]()[:numpy]()))
        model.blocks[i].mlp.fc2.b = Param(atype(weights["blocks.$(i-1).mlp.fc2.bias"][:cpu]()[:numpy]()))
        
        model.blocks[i].norm1.a = Param(atype(weights["blocks.$(i-1).norm1.weight"][:cpu]()[:numpy]()))
        model.blocks[i].norm1.b = Param(atype(weights["blocks.$(i-1).norm1.bias"][:cpu]()[:numpy]()))
        
        model.blocks[i].norm2.a = Param(atype(weights["blocks.$(i-1).norm2.weight"][:cpu]()[:numpy]()))
        model.blocks[i].norm2.b = Param(atype(weights["blocks.$(i-1).norm2.bias"][:cpu]()[:numpy]()))
   end

    # tokens_to_token
          # attn 1
    model.tokens_to_token.attention1.attn.proj.w = Param(atype(weights["tokens_to_token.attention1.attn.proj.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention1.attn.proj.b = Param(atype(weights["tokens_to_token.attention1.attn.proj.bias"][:cpu]()[:numpy]()))

    model.tokens_to_token.attention1.attn.qkv.w = Param(atype(weights["tokens_to_token.attention1.attn.qkv.weight"][:cpu]()[:numpy]()))

    model.tokens_to_token.attention1.mlp.fc1.w = Param(atype(weights["tokens_to_token.attention1.mlp.fc1.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention1.mlp.fc1.b = Param(atype(weights["tokens_to_token.attention1.mlp.fc1.bias"][:cpu]()[:numpy]()))
   
    model.tokens_to_token.attention1.mlp.fc2.w = Param(atype(weights["tokens_to_token.attention1.mlp.fc2.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention1.mlp.fc2.b = Param(atype(weights["tokens_to_token.attention1.mlp.fc2.bias"][:cpu]()[:numpy]()))

    model.tokens_to_token.attention1.norm1.a = Param(atype(weights["tokens_to_token.attention1.norm1.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention1.norm1.b = Param(atype(weights["tokens_to_token.attention1.norm1.bias"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention1.norm2.a = Param(atype(weights["tokens_to_token.attention1.norm2.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention1.norm2.b = Param(atype(weights["tokens_to_token.attention1.norm2.bias"][:cpu]()[:numpy]()))
    
          # attn2
    model.tokens_to_token.attention2.attn.proj.w = Param(atype(weights["tokens_to_token.attention2.attn.proj.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention2.attn.proj.b = Param(atype(weights["tokens_to_token.attention2.attn.proj.bias"][:cpu]()[:numpy]()))

    model.tokens_to_token.attention2.attn.qkv.w = Param(atype(weights["tokens_to_token.attention2.attn.qkv.weight"][:cpu]()[:numpy]()))

    model.tokens_to_token.attention2.mlp.fc1.w = Param(atype(weights["tokens_to_token.attention2.mlp.fc1.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention2.mlp.fc1.b = Param(atype(weights["tokens_to_token.attention2.mlp.fc1.bias"][:cpu]()[:numpy]()))
   
    model.tokens_to_token.attention2.mlp.fc2.w = Param(atype(weights["tokens_to_token.attention2.mlp.fc2.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention2.mlp.fc2.b = Param(atype(weights["tokens_to_token.attention2.mlp.fc2.bias"][:cpu]()[:numpy]()))

    model.tokens_to_token.attention2.norm1.a = Param(atype(weights["tokens_to_token.attention2.norm1.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention2.norm1.b = Param(atype(weights["tokens_to_token.attention2.norm1.bias"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention2.norm2.a = Param(atype(weights["tokens_to_token.attention2.norm2.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.attention2.norm2.b = Param(atype(weights["tokens_to_token.attention2.norm2.bias"][:cpu]()[:numpy]()))

         # project
    model.tokens_to_token.project.w = Param(atype(weights["tokens_to_token.project.weight"][:cpu]()[:numpy]()))
    model.tokens_to_token.project.b = Param(atype(weights["tokens_to_token.project.bias"][:cpu]()[:numpy]()))
    
    
    model.cls_token = Param(atype(weights["cls_token"][:cpu]()[:numpy]()))

    model.head.w = Param(atype(weights["head.weight"][:cpu]()[:numpy]()))
    model.head.b = Param(atype(weights["head.bias"][:cpu]()[:numpy]()))

    model.norm.a = Param(atype(weights["norm.weight"][:cpu]()[:numpy]()))
    model.norm.b = Param(atype(weights["norm.bias"][:cpu]()[:numpy]()))

    model.pos_embed = Param(atype(weights["pos_embed"][:cpu]()[:numpy]()))

    println("Pretrained weights are loaded successfully.")
    return model
end