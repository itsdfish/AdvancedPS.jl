module AdvancedPS
    using Random, StatsBase, ProgressMeter, Parameters, Distributions
    import AbstractMCMC: step!, AbstractSampler, AbstractModel
    import AbstractMCMC: bundle_samples, sample, MCMCThreads
    import MCMCChains: Chains
    export DE, Particle, DEModel, sample, MCMCThreads
    include("DEMCMC/structs.jl")
    include("DEMCMC/main.jl")
    include("DEMCMC/migration.jl")
    include("DEMCMC/crossover.jl")
    include("DEMCMC/mutation.jl")
    include("DEMCMC/utilities.jl")
end
