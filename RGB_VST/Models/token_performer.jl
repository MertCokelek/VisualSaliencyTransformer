export Token_performer

mutable struct Token_performer
    dim; in_dim; head_cnt; kernel_ratio; dp1; dp2;
    
    emb; kqv; dp; proj; norm1; norm2; epsilon; mlp; m; w;
    
    function Token_performer(dim, in_dim; head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1)
        self = new(dim, in_dim, head_cnt, kernel_ratio, dp1, dp2)
        self.emb = in_dim * head_cnt
        self.kqv = Linear(dim, 3*self.emb)
        self.dp = Dropout(dp1)
        # self.dp = dp1
        self.proj = Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        # self.norm1 = LayerNorm(dim) # forward pass
        # self.norm2 = LayerNorm(self.emb) "

        # TODO: gelu
        self.mlp = Sequential(
            Linear(self.emb, 1*self.emb),
            gelu,
            Linear(1*self.emb, self.emb),
            Dropout(dp2)
        )
        
        self.m = convert(Int32, self.emb * kernel_ratio)
        self.w = rand(self.m, self.emb)
        # TODO: Initialization method
        self.w = convert(KnetArray, self.w)
        return self
    end
end


function prm_exp(self, x)
    # x = (B, T, hs)
    # w = (m, hs)
    # return : x : B, T, m
    # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
    # therefore return exp(w^Tx - |x|/2)/sqrt(m)
    
    axis = length(size(x))
    xd = repeat(sum(x .* x, dims=axis), 1,1,self.m) ./ 2
    wtx = zeros(size(xd))
    @einsum wtx[b,t,m] = x[b,t,i] * w[m, i]
    return exp.(wtx - xd) / sqrt(self.m)
end

function single_attn(self, x)
    kqv_out = reshape(self.kqv(x), 3, self.emb)
    k, q, v = kqv_out[1,:], kqv_out[2,:], kqv_out[3,:]
    kq, qp = self.prm_exp(k), self.prm_exp(q) # B, T, m
    B, T, m = size(kq)
    D = zeros((B, T))
    kp_sum = sum(kp, dims=2)
    @einsum D[b,t] = qp[b,t,i] * kp_sum[b,i]
    D = reshape(D, (B, T, 1)) # (B, T, m) * (B, m) -> (B, T, 1)
    kptv = zeros((B, self.emb, m)); @einsum kptv[b,n,m] = v[b,i,n] * kp[b,i,m];
    y = zeros((B, T, self.emb));
    @einsum y[b,t,n] = qp[b,t,i] * kptv[b,n,i]
    y /= (repeat(D, 1, 1, self.emb) + self.epsilon)  # (B, T, emb)*Diag
    #skip connection
    # y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
    y = self.dp(self.proj(y))
    return y
end

function (self::Token_performer)(x)
    x += self.single_attn(LayerNorm(x))
    x += self.mlp(LayerNorm(x))
    return x
end