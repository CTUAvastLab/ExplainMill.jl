"""
    struct GradExplainer
        forward::Bool
    end

    Calculates importance of features as an absolute value of gradient with 
    respect to a mask set to all ones multiplying features / adjacency matrix. 
    This explanation was used in GNN explainer by Leskovec et al. in experimental
    section to show that their method is better.

    `forward` specifies, if we use ForwardMode or ReverseMode differentiation 
    (default is to use reverse, which is implemented through Zygote and suffers 
    terrible compilation times).
"""
struct GradExplainer
    forward::Bool
end

GradExplainer() = GradExplainer(false)

function stats(e::GradExplainer, ds, model, classes = onecold(model, ds), clustering = _nocluster)
    y = gnntarget(model, ds, classes)
    f(o) = sum(softmax(o) .* y)
    statsf(e, ds, model, f, clustering)
end

function statsf(::GradExplainer, ds, model, f, ::typeof(_nocluster))
    o = softmax(model(ds))
    mk = create_mask_structure(ds, d -> SimpleMask(ones(eltype(o), d)))
    ps = Flux.Params(map(m -> simplemask(m).x, collectmasks(mk)))
    gs = gradient(() -> f(model(ds, mk)), ps)
    mapmask(mk) do m, _
        d = length(m)
        if haskey(gs, m.x)
            HeuristicMask(abs.(gs[m.x])[:])
        else
            HeuristicMask(zeros(eltype(m.x), d))
        end
    end
end

@deprecate GradExplainer2 GradExplainer
