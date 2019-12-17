using Random
using Test
using Distributions
using AdvancedPS
using BenchmarkTools
using Libtask
const APS = AdvancedPS
using Turing
using Turing.Core: @varname
using Revise

dir = splitdir(splitdir(pathof(AdvancedPS))[1])[1]
push!(LOAD_PATH,dir*"/Example/Using_Turing_VI/" )
push!(LOAD_PATH,dir*"/Example/Using_Custom_VI/" )

using AdvancedPS_Turing_Container
const APSTCont = AdvancedPS_Turing_Container
import AdvancedPS_Turing_Container: tonamedtuple, merge_traj!, set_retained_vns_del_by_spl!
using AdvancedPS_SSM_Container
import AdvancedPS_SSM_Container: tonamedtuple, merge_traj!, set_retained_vns_del_by_spl!
const CustomCont= AdvancedPS_SSM_Container
include(dir*"/Tests/test_utils/AllUtils.jl")


include(dir*"/Tests/test_resample.jl")
include(dir*"/Tests/test_container.jl")
include(dir*"/Tests/Experiments/Custom_Container/tests.jl")
include(dir*"/Tests/Experiments/Turing_VI_Container/tests.jl")


include(dir*"/Tests/Using_Turing_VI/numerical_tests.jl")
include(dir*"/Tests/Using_Turing_VI/large_numerical_tests.jl")
include(dir*"/Tests/Using_Turing_VI/numerical_tests_pgas.jl")
include(dir*"/Tests/Using_Turing_VI/large_numerical_tests_pgas.jl")
# The other test do not make sense for this container...
include(dir*"/Tests/Using_Custom_Container/large_numerical_tests.jl")
include(dir*"/Tests/Using_Custom_Container/large_numerical_tests_pgas.jl")
