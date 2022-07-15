mutable struct Linear; 
    w; b;
    function Linear(input::Int,outputs...; bias=true)
        self = new(param(outputs...,input), bias ? param0(outputs...) : nothing)
        self.b = reshape(self.b, 1, 1, :)
        return self
    end
    
    function (l::Linear)(x)
        W1,W2,X1,X2 = size(l.w)[1:end-1], size(l.w)[end], size(x,3), size(x)[1:2]; 
        y = l.w * reshape(x,X1,:)
        y = reshape(y, X2..., W1...)
        if l.b !== nothing; y = y .+ l.b; end
        return y
    end
end






x = param(4, 3136, 64)
l = Linear(64, 128)

size(l(x))