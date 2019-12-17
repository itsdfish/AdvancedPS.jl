
@testset "apf.jl" begin
    @apf_testset "apf constructor" begin
        N = 200
        f = aps_gdemo_default
        model = PFModel(f, NamedTuple())
        tcontainer = VarInfo()

        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions(APSTCont.set_retained_vns_del_by_spl!,  APSTCont.tonamedtuple)
        sample(model, alg, uf, tcontainer, N)

        alg = AdvancedPS.PGAlgorithm(5)
        uf = AdvancedPS.PGUtilityFunctions(APSTCont.set_retained_vns_del_by_spl!,  APSTCont.tonamedtuple)
        sample(model, alg, uf, tcontainer, N)

    end
    @numerical_testset "apf inference" begin
        N = 5000
        f = aps_gdemo_default
        model = PFModel(f, NamedTuple())
        tcontainer = VarInfo()

        alg = AdvancedPS.PGAlgorithm(5)
        uf = AdvancedPS.PGUtilityFunctions( APSTCont.set_retained_vns_del_by_spl!,  APSTCont.tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_gdemo(caps1, atol = 0.2)

        alg = AdvancedPS.SMCAlgorithm()
        uf = AdvancedPS.SMCUtilityFunctions( APSTCont.set_retained_vns_del_by_spl!,  APSTCont.tonamedtuple)
        caps1 = sample(model, alg, uf, tcontainer, N)
        check_gdemo(caps1, atol = 0.2)
    end
end
