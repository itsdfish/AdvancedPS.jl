using AdvancedPS, Test
cd(@__DIR__)
println("Starting main test file")
println()
println()
println()
@testset "DE-MCMC Tests" begin
    include("DEMCMC_Tests.jl")
end
