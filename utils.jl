using Knet

export LayerNorm, Linear, Sequential, Dropout

mutable struct LayerNorm; a; b; ϵ; end

function LayerNorm(dmodel; eps=1e-6)
    a = param(dmodel; init=ones)
    b = param(dmodel; init=zeros)
    LayerNorm(a, b, eps)
end

function (l::LayerNorm)(x, o...)
    μ = mean(x,dims=1)
    σ = std(x,mean=μ,dims=1)
    ϵ = eltype(x)(l.ϵ)
    l.a .* (x .- μ) ./ (σ .+ ϵ) .+ l.b # To Do: doing x .- μ twice?
end


mutable struct Linear; w; b; end

function Linear(input::Int,outputs...; bias=true)
    Linear(param(outputs...,input), bias ? param0(outputs...) : nothing)
end

function (l::Linear)(x)
    W1,W2,X1,X2 = size(l.w)[1:end-1], size(l.w)[end], size(x,1), size(x)[2:end]; 
    @assert W2===X1
    y = reshape(l.w,:,W2) * reshape(x,X1,:)
    y = reshape(y, W1..., X2...)
    if l.b !== nothing; y = y .+ l.b; end
    return y
end

# Chain
mutable struct Sequential; layers; end

function Sequential(layer1, layer2, layers...)
    Sequential((layer1, layer2, layers...))
end

function (l::Sequential)(x, o...)
    for layer in l.layers
        x = layer(x, o...)
    end
    return x
end


# using Knet.Layers21: dropout
using Knet.Ops21: dropout

struct Dropout; p; end

function (l::Dropout)(x)        
    dropout(x, l.p)
end