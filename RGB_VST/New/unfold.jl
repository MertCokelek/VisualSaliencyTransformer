struct Unfold4D
    k; s; p;
end
    
function (self::Unfold4D)(x)
    b, c, h, w = size(x)
    
    k = self.k;
    p = self.p;
    s = self.s;
    
    hout = int((h + 2*p -k)/(s))+1
    wout = int((w + 2*p -k)/(s))+1

    y = param(KnetArray(zeros((b, hout*wout, k*k*c))))  # TODO param? KnetArray?

    x_pad = zeros((b, c, h+2p, w+2p))
    x_pad[:,:,p+1:end-p, p+1:end-p] .= x # .= ?
    
    h_offset = 0
    for i in (k÷2): s: h+p-1
        w_offset = 1
        for j in (k÷2):s: w+p-1
            rh1 = i+1-k÷2  # receptive field starting idx
            rh2 = i+1+k÷2  #+1 # ending idx
            rw1 = j+1-k÷2  # receptive field starting idx
            rw2 = j+1+k÷2  #+1 # ending idx
            y[:, h_offset*hout+w_offset, :, :] = reshape(x_pad[:, :, rh1:rh2, rw1:rw2], b, :)
            w_offset += 1
        end
        h_offset += 1
    end
    return y
end