include("t2t_vit.jl")
using Knet:KnetArray;

x = KnetArray(rand(4, 3, 224, 224));

tokens_to_token = T2T_module();

tokens_to_token(x)
