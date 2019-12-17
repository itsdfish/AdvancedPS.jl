#################################################################
#
# We reproduce the results form http://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf
# Experiment 1, page 2151
#
#################################################################
push!(LOAD_PATH, "/home/kongi/Julia-AdvancedPS/PGASAdvancedPS.jl/")
using Revise
using Distributions
using AdvancedPS
const APS = AdvancedPS
using BenchmarkTools
using Plots
dir = splitdir(splitdir(pathof(AdvancedPS))[1])[1]
push!(LOAD_PATH,dir*"/Example/Using_Custom_VI/" )
using AdvancedPS_SSM_Container
const APSCont = AdvancedPS_SSM_Container
#################################################################
#################################################################
# Nsamples
n = 101
# Amount of particles
# Note that we should not go higher because this makes the algorithm too slow
N = [5, 10, 20, 50, 100]
# Number of iterations
Niter = 100
#Hyper params form
a = 0.9
σv = 0.32
σe = 1.0
#################################################################
# We count the amount of updates!
updates_pgas = zeros(Int, n, length(N))
updates_pg = zeros(Int, n, length(N))
#################################################################
# Simulate a trajectory....
y = Array{Float64,2}(undef,1, n-1,)
xmdl = Array{Float64,2}(undef,1, n,)
xmdl[1,1] = rand(Normal(0.0, σv^2))

for i =1:n-1
    xmdl[1, i+1] = a*xmdl[1, i] + rand(Normal(0.0, σv^2))
    y[1, i] = xmdl[1, i+1] + rand(Normal(0.0, σe^2))
end

##################################################################
# The model is essentially the same...

function task_f(y)
    var = APSCont.initialize()
    x = zeros(1, n)
    r = rand(MultivariateNormal(1, σv^2))
    x[:, 1] = APSCont.update_var!(var, 1,  r)
    logp = logpdf(MultivariateNormal(1, σv^2), r)
    APSCont.report_transition!(var, logp, logp)
    for i = 2:n
        # Sampling
        r = rand(MultivariateNormal(a *x[:, i-1], ones(1,1)* σv^2))
        x[:, i] = APSCont.update_var!(var, i, r)
        logp = logpdf(MultivariateNormal(a *x[:, i-1], ones(1,1)* σv^2), x[:, i])
        APSCont.report_transition!(var,logp,logp)
        #Proposal and Resampling
        logpy = logpdf(MultivariateNormal(x[:, i], ones(1,1)* σe^2), y[: ,i-1])
        var = APSCont.report_observation!(var,logpy)
    end
end
#########################################################################################

tcontainer =  Container(zeros(1,n),Vector{Bool}(falses(n)),Vector{Int}(zeros(n)),0)
model = PFModel(task_f, (y=y,))
#########################################################################################
# First PGAS!
uf = AdvancedPS.PGASUtilityFunctions( APSCont.set_retained_vns_del_by_spl!, APSCont.tonamedtuple, APSCont.merge_traj!)
for j =1:length(N)
    alg = AdvancedPS.PGASAlgorithm(N[j])
    spl = Sampler(alg, uf, tcontainer)
    for i = 1:Niter+1
        n = spl.alg.n
        taskinfo = PGASTaskInfo(spl.alg.proposal_a_w)
        T = Trace{typeof(spl.vi), typeof(taskinfo)}
        if spl.ref_traj !== nothing
            particles = T[ APS.get_new_trace(spl.vi, model.task, taskinfo) for _ =1:n-1]
            pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n-1),0.0,0)
            # Reset Task
            spl.ref_traj = APS.forkr(spl.ref_traj)
            push!(pc, spl.ref_traj)
            APS.sample!(pc, spl.alg, spl.uf, spl.ref_traj)
            indx = AdvancedPS.randcat(weights(pc))
            new_particle = pc[indx]
            #Problem is just one dimensinoal, therefore, this is fine!
            updates_pgas[:,j] += vec(spl.ref_traj.vi.x .!= new_particle.vi.x)
            particle = spl.ref_traj
        else
            particles = T[ APS.get_new_trace(spl.vi, model.task, taskinfo) for _ =1:n]
            pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n),0.0,0)
            APS.sample!(pc, spl.alg, spl.uf, spl.ref_traj)
            indx = AdvancedPS.randcat(weights(pc))
            particle = spl.ref_traj = pc[indx]
        end
    end
    println("Done with PGAS $(N[j])")
end
###########################################################################################
#########################################################################################
# Then PG!
uf = AdvancedPS.PGUtilityFunctions( APSCont.set_retained_vns_del_by_spl!, APSCont.tonamedtuple)
for j =1:length(N)
    alg = AdvancedPS.PGAlgorithm(N[j])
    spl = APS.Sampler(alg, uf, tcontainer)
    for i = 1:Niter+1
        n = spl.alg.n
        taskinfo = PGTaskInfo()
        T = Trace{typeof(spl.vi), typeof(taskinfo)}
        if spl.ref_traj !== nothing
            particles = T[ APS.get_new_trace(spl.vi, model.task, taskinfo) for _ =1:n-1]
            pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n-1),0.0,0)
            # Reset Task
            spl.ref_traj = APS.forkr(spl.ref_traj)
            push!(pc, spl.ref_traj)
            APS.sample!(pc, spl.alg, spl.uf, spl.ref_traj)
            indx = AdvancedPS.randcat(weights(pc))
            new_particle = pc[indx]
            #Problem is just one dimensinoal, therefore, this is fine!
            updates_pg[:,j] += vec(spl.ref_traj.vi.x .!= new_particle.vi.x)
            particle = spl.ref_traj
        else
            particles = T[ APS.get_new_trace(spl.vi, model.task, taskinfo) for _ =1:n]
            pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n),0.0,0)
            APS.sample!(pc, spl.alg, spl.uf, spl.ref_traj)
            indx = AdvancedPS.randcat(weights(pc))
            particle = spl.ref_traj = pc[indx]
        end
    end
    println("Done with PG $(N[j])")
end
###########################################################################################

plotpgas = plot(1:n, updates_pgas./Niter, title="PGAS Update Rates", label=["$(N[j]) Particles" for j = 1:length(N)] )
savefig(plotpgas, "Update_Rates_PGAS.png")
plotpg = plot(1:n, updates_pg./Niter, title="PG Update Rates", label=["$(N[j]) Particles" for j = 1:length(N)] )
savefig(plotpg, "Update_Rates_PG.png")
