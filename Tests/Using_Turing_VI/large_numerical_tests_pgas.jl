
@testset "test_against_turing.jl" begin

    @numerical_testset "apf inference" begin
        N = 5000


        ## Testset1
        y = [0 for i in 1:10]

        # We use the Turing model as baseline
        chn_base = sample(large_demo(y),PG(20),N) # baselien

        #############################################
        # Using a Proposal                          #
        #############################################
        f = large_demo_apf_proposal
        model = PFModel(f, (y=y, ))
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


        # PG
        alg = AdvancedPS.PGASAlgorithm(20)
        uf = AdvancedPS.PGASUtilityFunctions( APSTCont.set_retained_vns_del_by_spl!,  APSTCont.tonamedtuple, APSTCont.merge_traj!)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base,0.1, 0.2)



        #############################################
        # No Proposal                               #
        #############################################
        f = large_demo_apf
        model = PFModel(f, (y=y, ))

        alg = AdvancedPS.PGASAlgorithm(20)
        uf = AdvancedPS.PGASUtilityFunctions( APSTCont.set_retained_vns_del_by_spl!,  APSTCont.tonamedtuple, APSTCont.merge_traj!)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base,0.1, 0.2)



        ## Testset2
        y = [-0.1*i for i in 1:10]

        # We use the Turing model as baseline
        chn_base = sample(large_demo(y),PG(20),N)

        #############################################
        # Using a Proposal                          #
        #############################################
        f = large_demo_apf_proposal
        model = PFModel(f, (y=y, ))

        # PG
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base, 0.1, 0.2)


        #############################################
        # No Proposal                               #
        #############################################
        f = large_demo_apf
        model = PFModel(f, (y=y, ))



        # PG
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical(caps1,chn_base, 0.1, 0.2)
    end
end
