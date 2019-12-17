
@testset "apf.jl" begin
    @apf_testset "apf constructor" begin
        N = 200
        f = aps_gdemo_default
        model = PFModel(f, NamedTuple())

        # PGAS requires the typed variant because of APSTCont.merge_traj!
        untypedcontainer = VarInfo()
        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(APSTCont.set_retained_vns_del_by_spl!, APSTCont.tonamedtuple)

        T = Trace{typeof(untypedcontainer),SMCTaskInfo{Float64}}
        particles = T[ APS.get_new_trace(untypedcontainer, model.task, SMCTaskInfo()) for _ =1:1]
        pc = ParticleContainer{typeof(particles[1])}(particles,zeros(1),0.0,0)
        AdvancedPS.sample!(pc, alg, uf, nothing)

        container = empty!(TypedVarInfo(pc[1].vi))
        tcontainer = container

        alg = AdvancedPS.PGASAlgorithm(5)
        uf = AdvancedPS.PGASUtilityFunctions(APSTCont.set_retained_vns_del_by_spl!,  APSTCont.tonamedtuple, APSTCont.merge_traj!)
        sample(model, alg, uf, tcontainer, N)

    end
    @numerical_testset "apf inference" begin
        N = 5000
        f = aps_gdemo_default
        model = PFModel(f, NamedTuple())


        # PGAS requires the typed variant because of APSTCont.merge_traj!
        untypedcontainer = VarInfo()
        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(APSTCont.set_retained_vns_del_by_spl!, APSTCont.tonamedtuple)

        T = Trace{typeof(untypedcontainer),SMCTaskInfo{Float64}}
        particles = T[ APS.get_new_trace(untypedcontainer, model.task, SMCTaskInfo()) for _ =1:1]
        pc = ParticleContainer{typeof(particles[1])}(particles,zeros(1),0.0,0)
        AdvancedPS.sample!(pc, alg, uf, nothing)

        container = empty!(TypedVarInfo(pc[1].vi))
        tcontainer = container


        alg = AdvancedPS.PGASAlgorithm(5)
        uf = AdvancedPS.PGASUtilityFunctions( APSTCont.set_retained_vns_del_by_spl!,  APSTCont.tonamedtuple, APSTCont.merge_traj!)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_gdemo(caps1, atol = 0.2)

    end
end
