using Knet:param, param0, atype
using Statistics: mean, std
int(x) = floor(Int8,x)
Tensor(x) = convert(KnetArray, x)


mutable struct LayerNorm; a; b; ϵ; end

function LayerNorm(dmodel; eps=1e-6)
    dim = (1, 1, dmodel)
    a = param(dim; init=ones)
    b = param(dim; init=zeros)
    ln = LayerNorm(a, b, eps)
end

function (l::LayerNorm)(x, o...)
    newdim = (1, 1, maximum(size(l.a)))
    l.a = reshape(l.a, newdim)
    l.b = reshape(l.b, newdim)
    μ = mean(x,dims=(2, 3))
    σ = std(x,mean=μ,dims=(2, 3))
    ϵ = eltype(x)(l.ϵ)
    x_norm = (x .- μ) ./ (σ .+ ϵ)
    
    l.a .* x_norm .+ l.b # TODO: doing x .- μ twice?
end



# struct Unfold
#     k; s; p;
# end
    
# function (self::Unfold)(T)
#     println("Before Unfold")
#     @show size(T)
    
#     T = permutedims(T, (1,3,4,2))
#     b, h, w, c = size(T)
    
#     k = self.k;
#     p = self.p;
#     s = self.s;
#     hout = int((h + 2*p -k)/(s))+1
#     wout = int((w + 2*p -k)/(s))+1

#     Tout = zeros((b, k*k*c, hout*wout))

#     for batch_i in 1:b
        
#         Tpadded = zeros((h+2p, w+2p, c))
#         Tpadded[p+1:end-p, p+1:end-p,:] .= T[batch_i]

#         ctr_i = 0
#         for i in (k÷2): s: h+p
#             ctr_j = 1

#             for j in (k÷2):s: w+p
#                 rh1 = i+1-k÷2  # receptive field starting idx
#                 rh2 = i+1+k÷2#+1 # ending idx
#                 rw1 = j+1-k÷2  # receptive field starting idx
#                 rw2 = j+1+k÷2#+1 # ending idx

#                 receptive_field = Tpadded[rh1:rh2, rw1:rw2,:1]
#                 receptive_field = receptive_field'

#                 receptive_field = reduce(vcat, receptive_field)

#                 receptive_field_pad = zeros((k*k*3))
#                 sz = size(receptive_field)[1]
#                 receptive_field_pad[1:sz] .= receptive_field
#                 Tout[batch_i, :, ctr_i*hout+ctr_j] = receptive_field_pad
#                 ctr_j += 1
#             end
#             ctr_i += 1
#         end
#         println("After Unfold")
#         @show size(Tout)
#         return Tout
#     end

# end

using Knet.Ops21: dropout

struct Dropout; p; end

function (l::Dropout)(x)        
    dropout(x, l.p)
end

struct get_sinusoid_encoding 
    w 
end

function get_sinusoid_encoding(n_position, d_hid; λ=10000, atype=atype())
    x = exp.((0:2:d_hid-1) .* -(log(λ)/d_hid)) * (0:n_position-1)'
    pe = zeros(d_hid, n_position)
    pe[1:2:end,:] = sin.(x)
    pe[2:2:end,:] = cos.(x)
    get_sinusoid_encoding(atype(pe))
end

function (l::get_sinusoid_encoding)(x)
    x .+ l.w[:,1:size(x,2)]
end