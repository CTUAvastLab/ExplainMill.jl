```@raw html
<img class="display-light-only" src="assets/logo.svg" alt="ExplainMill.jl logo" style="width: 40%;"/>
<img class="display-dark-only" src="assets/logo-dark.svg" alt="ExplainMill.jl logo" /style="width: 40%;">
```

## Installation

Run the following in REPL:

```julia
] add ExplainMill
```

Julia v1.9 or later is required.

## Getting started

* [References](@ref): related literature
* [Citation](@ref): preferred citation entries


# Breaking change
As discussed below, I have removed the "feature" that dict with single child is ignored. 

Below snippet fix cuckoo model and data trained on schema created by  JsonGrinder version 1.6.1 and reasonably below.

```
function fixmodel(model)
	mm = @set model.ms.behavior.ms.enhanced.im = ProductModel((data = model[:behavior][:enhanced].im,), identity)
	mm = @set mm.ms.clipboard_changes.im = ProductModel((to = mm[:clipboard_changes].im,), identity)
	mm = @set mm.ms.signatures.im = ProductModel((name = mm[:signatures].im,), identity)
	mm = @set mm.ms.network.ms.domains.im = ProductModel((domain = mm[:network][:domains].im,), identity)
end

function fixdata(ds)
	dd = @set ds.data.behavior.data.enhanced.data = ProductNode((data = ds[:behavior][:enhanced].data,))
	dd = @set dd.data.clipboard_changes.data = ProductNode((to = dd[:clipboard_changes].data,))
	dd = @set dd.data.signatures.data = ProductNode((name = dd[:signatures].data,))
	dd = @set dd.data.network.data.domains.data = ProductNode((domain = dd[:network][:domains].data,))
end
```

# Discussion

* Should we explicitly model missing in categorical variable as an n + 2 item?
* Remove skipping of Dictionary in JsonGrinder and replace it with the IdentityModel()
* ArrayModel in Mill should have a default for missing values and potentially ProductModel

# ExplainMill.jl

This library provides an explanation of hierarchical multi-instance learning
models using the method of Shapley values. This explanation method works by
randomly perturbing "features" of the sample and observing the output of the
classifier. The rationale behind the method is that if the feature is
non-informative than the output of the classifier should be non-sensitive
to its perturbation (and vice-versa). To support hierarchical models,
we define several types of "daf" explainers.
* `BagDaf` explains `Mill.BagNode` by removing instances
* `TreeDaf` is just a container and does not explain anything
* `ArrayDaf` explains dense matrices (`Mill.ArrayNode{Matrix}`) by removing
features of all samples (instances). Explaining individual items in dense
matrices can be easily added, but at the moment author (Pevnak) is more interested in the influence of a particular feature on the sample. Features are modified by setting them to zero. This method assumes that features are centred, which is considering the state of JsonGrinder unlikely. A better method would be to provide a set samples such that values of modified features are replaced with a random sample from the population.
* `SparseArrayDaf` explains sparse matrix by zeroing individual items of the
random matrix.

Let's demonstrate the usage. Assume that we want to explain sample `ds` with
model `model` using 1000 random samples. After necessary setup the important
line is `ExplainMill.explain(ds, model, n = 1000)` which does the explanation

```
using Mill, ExplainMill, SparseArrays, Flux

an = ArrayNode(randn(2,5))
cn = ArrayNode(sprand(2,5, 0.5))
bn = BagNode(ProductNode((a = an, c = cn)), AlignedBags([1:2,3:5]))
tn = ProductNode((a = an[1:2], b = bn))
ds = BagNode(tn, AlignedBags([1:2]))
model = reflectinmodel(ds, d -> Dense(d,2),
		d -> SegmentedMean(d),
		b = Dict("" => d -> Dense(d,1)))

explained_ds = ExplainMill.explain(ds, model, n = 1000)
```

The output of the model has to have dimension one. You might find convenient
to use the following code to select specific output

```
using Setfield
new_model = @set model.bm.m = Chain(model.bm, x -> x[1,:])
```

### Peaking under the hood
The explanation starts by creating a structure holding a shapley values for
individual nodes. The structure resembles the Mill structure, but instead of
data it holds `Duff.Daf` statistics. For a `ds`, the structure is created as

```
daf = ExplainMill.Daf(ds)
```

then, in the loop we repeatedlu create a sub-sample, calculate the shapley value,
and update the stats

```
using StatsBase, Duff
dss, mask = sample(daf, ds)
v = ExplainMill.scalar(model(dss))
Duff.update!(daf, mask, v)
```

once the sufficient statistic is calculated, we extract the `mask` which will
allow to prune the sample (by default everything all true which means no
pruning.) Masks from individual level is returned together with daf statistics
as a second argument, which allows to easily link daf statistics to
cooresponding items in mask and in hierarchical mask. Using `CatViews` package
further simplifies the construction, as

```
using CatViews
mask, dafs = ExplainMill.masks_and_stats(daf)
catmask = CatView(tuple([d.m for d in dafs]...))
pvalue = CatView(tuple([Duff.UnequalVarianceTTest(d.d) for d in dafs]...))
```

`catmask` now behaves like a single array and similarly the p-values of a test
where means of samples when the item / feaure is present. This means and if
if we change item in `catmask`, the change is propagated to `mask` and
`prune(ds, mask)` return the sample without the corresponding item. Note that
since statistics is effectively zero as it has not been updated, pvalues are NaNs

Adding a support for new type of nodes (lists) is currently involved and it
might undergo further changes. At the moment, it is simple to take it out of
the explanation. For example if we want to remove PathNode from explanation,
define constructor as `Duff.Daf(::PathNode) = ExplainMill.SkipDaf()`

## Design thoughts on logic output

* We need to carry observations that will be exported downward, otherwise skipped explanation and export of arrays would not work properly. This can get into interesting situations, where 
- something can be present because of the upper mask but missing because of the lower mask. In this case, I will emit missing
