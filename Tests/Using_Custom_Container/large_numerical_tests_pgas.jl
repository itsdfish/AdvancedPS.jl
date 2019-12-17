
@testset "test_against_turing.jl" begin

    @numerical_testset "apf inference" begin
        N = 5000
        T = 11


        ## Testset1
        y = [0 for i in 1:10]

        # We use the Turing model as baseline
        chn_base = sample(large_demo(y),PG(20),5*N) # baselien

        #############################################
        # Using a Proposal                          #
        #############################################
        f = large_demo_apf_proposal_c
        model = PFModel(f, (y=y, ))
        # PGAS requires the typed variant because of merge_traj!
        tcontainer =  Container(zeros(1, T),Vector{Bool}(falses(T)),Vector{Int}(zeros(T)),0)
        # PG
        alg = AdvancedPS.PGASAlgorithm(20)
        uf = AdvancedPS.PGASUtilityFunctions(CustomCont.set_retained_vns_del_by_spl!,  CustomCont.tonamedtuple, CustomCont.merge_traj!)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical_custom(caps1,chn_base, T, 0.1, 0.2)

        #############################################
        # No Proposal                               #
        #############################################
        f = large_demo_apf_c
        model = PFModel(f, (y=y, ))
        # PGAS requires the typed variant because of merge_traj!

        # PG
        alg = AdvancedPS.PGASAlgorithm(20)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical_custom(caps1, chn_base, T, 0.1, 0.2)


        ## Testset2
        y = [-0.1*i for i in 1:10]

        # We use the Turing model as baseline
        chn_base = sample(large_demo(y),PG(20),N)

        #############################################
        # Using a Proposal                          #
        #############################################
        f = large_demo_apf_proposal_c
        model = PFModel(f, (y=y, ))
        # PGAS requires the typed variant because of merge_traj!


        # PG
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical_custom(caps1, chn_base, T, 0.1, 0.2)

        #############################################
        # No Proposal                               #
        #############################################
        f = large_demo_apf_c
        model = PFModel(f, (y=y, ))


        # PG
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_numerical_custom(caps1,chn_base, 0.1, 0.2)
    end
end
