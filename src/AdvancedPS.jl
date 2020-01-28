module AdvancedPS
    using Random, AbstractMCMC, StatsBase, ProgressMeter, Parameters, Distributions
    import AbstractMCMC: step!, AbstractSampler, AbstractTransition, transition_type
    import AbstractMCMC: bundle_samples, sample, psample, AbstractModel
    import MCMCChains: Chains
    export DE, Particle, Model, sample, psample
    include("DEMCMC/structs.jl")
    include("DEMCMC/main.jl")
    include("DEMCMC/migration.jl")
    include("DEMCMC/crossover.jl")
    include("DEMCMC/mutation.jl")
    include("DEMCMC/utilities.jl")
end
