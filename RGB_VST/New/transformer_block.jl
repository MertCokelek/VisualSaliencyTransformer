mutable struct Linear; 
    w; b;
    function Linear(input::Int,outputs...; bias=true)
        self = new(param(outputs...,input), bias ? param0(outputs...) : nothing)
        if bias; self.b = reshape(self.b, 1, 1, :);end
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


mutable struct Mlp
    in_features; hidden_features; out_features; act_layer; drop_rate;
    
    drop; fc1; act; fc2;
    
    function Mlp(in_features; hidden_features=nothing, out_features=nothing, act_layer::Function=gelu, drop_rate=0.)
        self = new(in_features, hidden_features, out_features, act_layer, drop_rate)
        
        out_features = out_features === nothing ? in_features : out_features
        hidden_features = hidden_features === nothing ? in_features : hidden_features
        
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer#() ????
        self.fc2 = Linear(hidden_features, out_features)
        # self.drop = drop # called in forward pass
        self.drop = Dropout(drop_rate)
        return self
    end
    
    function (self::Mlp)(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x
    end
end