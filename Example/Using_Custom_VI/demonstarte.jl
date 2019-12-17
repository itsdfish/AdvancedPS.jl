
## It is not yet a package...
push!(LOAD_PATH, "/home/kongi/Julia-AdvancedPS/PGASAdvancedPS.jl/")

using Revise

using Distributions
using AdvancedPS
using BenchmarkTools


dir = splitdir(splitdir(pathof(AdvancedPS))[1])[1]
push!(LOAD_PATH,dir*"/Example/Using_Custom_VI/" )
using AdvancedPS_SSM_Container
const APSCont = AdvancedPS_SSM_Container

# Define a short model.
# The syntax is rather simple. Observations need to be reported with report_observation.
# Transitions must be reported using report_transition.
# The trace contains the variables which we want to infer using particle gibbs.
# Thats all!
n = 20


y = Array{Float64,2}(undef,1 ,n-1)
for i =1:n-1
    y[1 ,i] = 0
end



function task_f(y)
    var = APSCont.initialize()
    x = zeros(1, n)
    r = rand(MultivariateNormal(1,1.0))
    x[:, 1] = APSCont.update_var!(var, 1,  r)
    APSCont.report_transition!(var,0.0,0.0)
    for i = 2:n
        # Sampling
        r = rand(MultivariateNormal(1,1.0))
        x[:, i] = APSCont.update_var!(var, i, r)
        logγ = logpdf(MultivariateNormal(1,1.0),x[:, i]) #γ(x_t|x_t-1)
        logp = logpdf(MultivariateNormal(1,1.0),x[:, i])                # p(x_t|x_t-1)
        APSCont.report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(MultivariateNormal(x[:, i],1.0), y[: ,i-1])
        var = APSCont.report_observation!(var,logpy)
    end
end

tcontainer =  Container(zeros(1, n),Vector{Bool}(falses(n)),Vector{Int}(zeros(n)),0)
model = PFModel(task_f, (y=y,))



alg = AdvancedPS.SMCAlgorithm()
uf = AdvancedPS.SMCUtilityFunctions(APSCont.set_retained_vns_del_by_spl!, APSCont.tonamedtuple)
chn1 = sample(model, alg, uf, tcontainer, 10)

alg = AdvancedPS.PGAlgorithm(10)
uf = AdvancedPS.SMCUtilityFunctions(APSCont.set_retained_vns_del_by_spl!, APSCont.tonamedtuple)
chn1 = sample(model, alg, uf, tcontainer, 10)


alg = AdvancedPS.PGASAlgorithm(10)
uf = AdvancedPS.PGASUtilityFunctions( APSCont.set_retained_vns_del_by_spl!, APSCont.tonamedtuple, APSCont.merge_traj!)
@elapsed chn2 = sample(model, alg, uf, tcontainer, 10)

chn2
