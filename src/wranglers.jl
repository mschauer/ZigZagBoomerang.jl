@inline @generated function _action_nextaction(wrangler::FunctionWrangler{TOp, TNext}, next_action, Q, action, idx, args...) where {TNext, TOp}
    argnames = [:(args[$i]) for i = 1:length(args)]
    return quote
        if idx == 1
            affected = wrangler.op($(argnames...))
            for j in affected 
                #τ, e = sfindmin(next_action, j, $(argnames...))
                τ, e = _sfindmin(next_action, Inf, 0, 1, j, $(argnames...))
                Q[j] = τ
                action[j] = e   
            end
        else
            return _action_nextaction(wrangler.next, next_action, Q, action, idx - 1, $(argnames...))
        end
    end
end

"""
    sindex(wrangler::FunctionWrangler, idx, args...)
Call the `idx`-th function with args.
Note that this call iterates the wrangler from 1 to `idx`. Try to
put the frequently called functions at the beginning to minimize overhead.
"""
function action_nextaction(wrangler::FunctionWrangler, next_action, Q, action, idx, args...) 
    @assert idx <= length(wrangler)
    _action_nextaction(wrangler, next_action, Q, action, idx, args...)
end

@inline @generated function _sfindmin(wrangler::FunctionWrangler{TOp, TNext}, tmin, j, n, args...) where {TNext, TOp}
    TOp === Nothing && return :(tmin, j)
    argnames = [:(args[$i]) for i = 1:length(args)]
    return quote
        t = wrangler.op($(argnames...))
        tmin, j = t ≤ tmin ? (t, n) : (tmin, j)
        return _sfindmin(wrangler.next, tmin, j, n + 1, $(argnames...))            
    end
end

"""
    sfindmin(wrangler::FunctionWrangler, args...)
    
Look for the function which returns smallest value for the given arguments, and returns its index.
"""
sfindmin(wrangler::FunctionWrangler, args...) = _sfindmin(wrangler, Inf, 0, 1, args...)
