using Knet: param, KnetArray

# export Unfold, LayerNorm, Linear, Sequential, Dropout
export Unfold

int(x) = floor(Int8,x)
Tensor(x) = convert(KnetArray, x)

mutable struct Unfold
    k; s; p;
end
    
# function Unfold(kernel_size, stride, padding)
#     Unfold(kernel_size, stride, padding);
#     return self;
# end
function (self::Unfold)(T)
    T = permutedims(T, (1,3,4,2))
    b, h, w, c = size(T)
    # b, c, h, w = size(T)
    
    k = self.k;
    p = self.p;
    s = self.s;
    hout = int((h + 2*p -k)/(s))+1
    wout = int((w + 2*p -k)/(s))+1

    Tout = zeros((b, k*k*c, hout*wout))

    for batch_i in 1:b
        
        Tpadded = zeros((h+2p, w+2p, c))
        Tpadded[p+1:end-p, p+1:end-p,:] .= T[batch_i]

        ctr_i = 0
        for i in (k÷2): s: h+p
            ctr_j = 1

            for j in (k÷2):s: w+p
                rh1 = i+1-k÷2  # receptive field starting idx
                rh2 = i+1+k÷2#+1 # ending idx
                rw1 = j+1-k÷2  # receptive field starting idx
                rw2 = j+1+k÷2#+1 # ending idx
                # @show rh1, rh2, rw1, rw2

                receptive_field = Tpadded[rh1:rh2, rw1:rw2,:1]
                receptive_field = receptive_field'

                receptive_field = reduce(vcat, receptive_field)

                receptive_field_pad = zeros((k*k*3))
                sz = size(receptive_field)[1]
                # @show size(receptive_field), size(receptive_field_pad)
                receptive_field_pad[1:sz] .= receptive_field
                Tout[batch_i, :, ctr_i*hout+ctr_j] = receptive_field_pad
                ctr_j += 1
            end
            ctr_i += 1
        end
        return Tout
    end
end

mutable struct LayerNorm; a; b; ϵ; dim; end

function LayerNorm(dmodel; eps=1e-6)
    a = param(dmodel; init=ones)
    b = param(dmodel; init=zeros)
    LayerNorm(a, b, eps, (dmodel, 1, 1))
end

function (l::LayerNorm)(x, o...)
    # l.a = reshape(l.a, l.dim)
    # l.b = reshape(l.b, l.dim)
    μ = mean(x,dims=(2, 3))
    σ = std(x,mean=μ,dims=(2, 3))
    ϵ = eltype(x)(l.ϵ)    
    @show size(l.a), size(l.b), size(x)
    @show typeof(l.a), typeof(l.b), typeof(x)
    @show size(μ)
    l.a .* (x .- μ) ./ (σ .+ ϵ) .+ l.b # To Do: doing x .- μ twice?
end


mutable struct Linear; w; b; end

function Linear(input::Int,outputs...; bias=true)
    Linear(param(outputs...,input), bias ? param0(outputs...) : nothing)
end

function (l::Linear)(x)
    @show l
    W1,W2,X1,X2 = size(l.w)[1:end-1], size(l.w)[end], size(x,1), size(x)[2:end]; 
    @show W1, W2, X1, X2
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