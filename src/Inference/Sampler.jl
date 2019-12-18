

# The particle container is our state,

mutable struct SMCSampler{PC, ALG, UF, C} <: AbstractPFSampler where {
    PC<:ParticleContainer,
    ALG<:SMCAlgorithm,
    UF<:SMCUtilityFunctions
}
    pc        :: PC
    alg       :: ALG
    uf        :: UF
    vi        :: C
end


function Sampler(alg:: ALG, uf::UF, vi::C) where {
    C,
    ALG<: SMCAlgorithm,
    UF<: SMCUtilityFunctions,
}
    pc = ParticleContainer(Trace{typeof(vi),SMCTaskInfo{Float64}}[])
    SMCSampler(pc, alg, uf, vi)
end




mutable struct PGSampler{T, ALG, UF, C} <: AbstractPGSampler where {
    T <:Particle,
    ALG<:PGAlgorithm,
    UF<:PGUtilityFunctions
}
    alg       :: ALG
    uf        :: UF
    ref_traj  :: Union{T, Nothing}
    vi        :: C
end


Sampler(alg:: ALG, uf::UF, vi::T) where {
    T,
    ALG<: Union{PGAlgorithm},
    UF<: PGUtilityFunctions
} = PGSampler{Trace{typeof(vi),PGTaskInfo{Float64}},typeof(alg),typeof(uf),typeof(vi)}(alg, uf, nothing, vi)

mutable struct PGASSampler{T, ALG, UF, C} <: AbstractPGSampler where {
    T <:Particle,
    ALG<:PGASAlgorithm,
    UF<:PGASUtilityFunctions
}
    alg       :: ALG
    uf        :: UF
    ref_traj  :: Union{T, Nothing}
    vi        :: C
end




function Sampler(alg::ALG, uf::UF, vi::T) where {
    T,
    ALG<: Union{PGASAlgorithm},
    UF<: PGASUtilityFunctions
}
    taskinfo = PGASTaskInfo(alg.proposal_a_w)
    PGASSampler{Trace{typeof(vi),typeof(taskinfo)},typeof(alg),typeof(uf),typeof(vi)}(alg, uf, nothing, vi)
end


mutable struct PGASSamplerFullStates{T, ALG, UF, C} <: AbstractPGSampler where {
    T<:       Particle,
    ALG<:     PGASAlgorithm,
    UF<:      PGASUtilityFunctions,
    PC<:      ParticleContainer
}
    alg       :: ALG
    uf        :: UF
    ref_traj  :: Union{T, Nothing}
    vi        :: C
    pc        :: PC
end

function Sampler(alg::ALG, uf::UF, vi::T) where {
    T,
    ALG<: Union{PGASFullStatesAlgorithm},
    UF<: PGASUtilityFunctions
}
    taskinfo = PGASTaskInfo(alg.proposal_a_w)
    pc = ParticleContainer(Trace{typeof(vi),typeof(taskinfo)}[])
    PGASSamplerFullStates{Trace{typeof(vi),typeof(taskinfo)},typeof(alg),typeof(uf),typeof(vi)}(alg, uf, nothing, vi, pc)
end
