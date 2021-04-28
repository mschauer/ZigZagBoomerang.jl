

@inline @generated function _switch(i, f, g, q, args...)
    argnames = [:(args[$i]) for i = 1:length(args)]
    n = length(f.parameters)
    qu = quote 
        if i == $n
            uJ = f[$n]($(argnames...))
            for j in uJ[2]
                τ, e = _sfindmin(g, Inf, 0, 1, j, $(argnames...))
                q[1][j] = τ
                q[2][j] = e   
            end
            uJ[1]
        else
            throw(BoundsError())
        end
    end
    for k in n-1:-1:1
        qu = quote 
            if i == $k
                uJ = f[$k]($(argnames...))
                for j in uJ[2]
                    τ, e = _sfindmin(g, Inf, 0, 1, j, $(argnames...))
                    q[1][j] = τ
                    q[2][j] = e   
                end
                uJ[1]
            else $qu
            end
        end
    end
    return qu
end
switch(i, f, g, q, args...) = _switch(i, f, g, q, args...)
#time switch(2, (+, -), nothing, nothing, 1, 2)


@inline @generated function _switch1(i, f, args...)
    argnames = [:(args[$i]) for i = 1:length(args)]
    n = length(f.parameters)
    qu = quote 
        if i == $n
            f[$n]($(argnames...))
        else
            BoundsError()
        end
    end
    for k in n-1:-1:1
        qu = quote 
            if i == $k
                f[$k]($(argnames...))
            else $qu
            end
        end
    end
    return qu
end
switch1(i, f, args...) = _switch1(i, f, args...)
