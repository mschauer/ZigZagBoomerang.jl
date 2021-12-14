function marginal_sticky(i, tr)
    x = tr[:, i]
    N = length(x)
    res = 0.0
    for xi in x
        if abs(xi) <= eps() # numerical 
            res += 1/N
        end
    end
    res  
end


function el_conf_matrix(xitr, xis)
    if xitr != 0.0 # "positive" 
        if xis != 0.0
            return 4 # true positive
        else
            return 2 #
        end
    else # xitr == 0.0 "negative"
        if xis == 0.0
            return 1 # true negative
        else
            return 3
        end
    end
end


function inclusion_prob(trace, w, xtrue)
    d = size(trace,2)
    N = size(trace, 1)
    st = [marginal_sticky(i, trace, N) for i in 1:d]
    ip = sortperm(st)
    t0 = round(Int, w*d) # number of true 0
    pred = zeros(d)
    [i > t0 ?  pred[i] = 0.0 : pred[i] = 1.0 for i in ip]
    conf_matrix = zeros(d)
    [conf_matrix[i] = el_conf_matrix(xtrue[i], pred[i]) for i in 1:d]
    return conf_matrix
end




