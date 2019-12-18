
#SMC step
function AbstractMCMC.step!(
    ::AbstractRNG,
    model::AbstractPFModel,
    spl::SPL,
    ::Integer;
    iteration::Integer,
    kwargs...
    ) where SPL <: SMCSampler

    particle = spl.pc[iteration]

    params = spl.uf.tonamedtuple(particle.vi)
    return PFTransition(params, particle.taskinfo.logp, spl.pc.logE, weights(spl.pc)[iteration])
end


# PG step
function AbstractMCMC.step!(
    ::AbstractRNG,
    model::AbstractPFModel,
    spl::SPL,
    ::Integer;
    kwargs...
    ) where SPL <:Union{PGSampler}

    n = spl.alg.n

    T = Trace{typeof(spl.vi), PGTaskInfo{Float64}}

    if spl.ref_traj !== nothing
        particles = T[ get_new_trace(spl.vi, model.task, PGTaskInfo()) for _ =1:n-1]
        pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n-1),0.0,0)
        # Reset Task
        spl.ref_traj = forkr(spl.ref_traj)
        push!(pc, spl.ref_traj)
    else
        particles = T[ get_new_trace(spl.vi, model.task, PGTaskInfo()) for _ =1:n]
        pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n),0.0,0)
    end

    sample!(pc, spl.alg, spl.uf, spl.ref_traj)

    indx = AdvancedPS.randcat(weights(pc))
    particle = spl.ref_traj = pc[indx]
    params = spl.uf.tonamedtuple(particle.vi)
    return PFTransition(params, particle.taskinfo.logp, pc.logE, weights(pc)[indx])
end

function AbstractMCMC.step!(
    ::AbstractRNG,
    model::AbstractPFModel,
    spl::SPL,
    ::Integer;
    kwargs...
    ) where SPL <:Union{PGASSampler, PGASFullStatesAlgorithm}

    n = spl.alg.n
    taskinfo = PGASTaskInfo(spl.alg.proposal_a_w)
    T = Trace{typeof(spl.vi), typeof(taskinfo)}
    if spl.ref_traj !== nothing
       particles = T[ get_new_trace(spl.vi, model.task, taskinfo) for _ =1:n-1]
       pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n-1),0.0,0)
       # Reset Task
       spl.ref_traj = forkr(spl.ref_traj)
       push!(pc, spl.ref_traj)
    else
       particles = T[ get_new_trace(spl.vi, model.task, taskinfo) for _ =1:n]
       pc = ParticleContainer{typeof(particles[1])}(particles,zeros(n),0.0,0)
    end

    sample!(pc, spl.alg, spl.uf, spl.ref_traj)

    indx = AdvancedPS.randcat(weights(pc))
    particle = spl.ref_traj = pc[indx]
    if typeof(spl)<:PGASFullStatesAlgorithm
       spl.pc = pc
    end
    params = spl.uf.tonamedtuple(particle.vi)
    return PFTransition(params, particle.taskinfo.logp, pc.logE, weights(pc)[indx])
end
