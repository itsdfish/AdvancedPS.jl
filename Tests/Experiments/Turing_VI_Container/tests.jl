



@testset "Custom_Container.jl" begin
    @apf_testset "merge_traj" begin

        n = Ref(0)
        alg = AdvancedPS.PGASAlgorithm(5)
        uf = AdvancedPS.PGASUtilityFunctions(APSTCont.set_retained_vns_del_by_spl!,  APSTCont.tonamedtuple, APSTCont.merge_traj!)
        utcontainer = VarInfo()

        dist = Normal(0, 1)

        function fpc()
            var = APSTCont.initialize()
            i = 0
            for k = 1:30
                r = rand(dist)
                i += 1
                vn = @varname x[i]
                APSTCont.update_var!(var, vn, r)
                produce(0)
                var = current_trace()
                r = rand(dist)
                i += 1
                vn = @varname x[i]
                APSTCont.update_var!(var, vn, r)
                produce(1)
                var = current_trace()
            end
            produce(Val(:done))
        end

        # We want type stability!
        model = PFModel(fpc, NamedTuple())
        particles = [AdvancedPS.get_new_trace(utcontainer, model.task, APS.PGASTaskInfo(0.0, 0.0, 0.0)) for _ in 1:1]
        pc = ParticleContainer(particles)
        while consume(pc[1]) != Val(:done) end
        tcontainer = Turing.Core.RandomVariables.empty!(VarInfo{<:NamedTuple}(pc[1].vi))

        #Now lets go...

        model = PFModel(fpc, NamedTuple())
        particles = [AdvancedPS.get_new_trace(tcontainer, model.task, APS.PGASTaskInfo(0.0, 0.0, 0.0)) for _ in 1:1]
        pc = ParticleContainer(particles)
        while consume(pc[1]) != Val(:done) end

        ref_traj = APS.forkr(pc[1])
        ref_traj.vi.metadata.x.orders
        model = PFModel(fpc, NamedTuple())
        particles = [AdvancedPS.get_new_trace(tcontainer, model.task, APS.PGASTaskInfo(0.0, 0.0, 0.0)) for _ in 1:2]

        pc = ParticleContainer(particles)
        pc = push!(pc, ref_traj)

        base_ref = copy(ref_traj)


        consume(pc) == log(1)



        @test sum(base_ref.vi.metadata.x.vals .!= ref_traj.vi.metadata.x.vals) == 0

        indx = [2,3]
        pold_2 = copy(pc[2])
        push!(indx,length(pc))
        resample!(pc, uf, indx, ref_traj, nothing)
        @test pc[2].vi.metadata.x.flags["del"][1:2] == BitArray([0, 1])
        @test pc[3] === ref_traj
        @test sum(pc[1].vi.metadata.x.vals .!= pold_2.vi.metadata.x.vals) == 0
        # Tests set_retained_vns_del_by_spl
        @test pc[1].vi.x.flags["del"] == BitArray([0])
        @test sum(pc[2].vi.x.flags["del"]) == 59
        @test sum(pc[3].vi.x.flags["del"] .= 0) == 0

        ancestor_index = 1
        # We are one step behind....
        selected_path = pc[ancestor_index]
        new_vi = uf.merge_traj!(copy(selected_path.vi), ref_traj.vi) # Merge trajectories
        @test sum(new_vi.metadata.x.flags["del"]) == 0

        @test  sum(new_vi.metadata.x.vals[1] .!= selected_path.vi.metadata.x.vals[1]) == 0
        @test  sum(new_vi.metadata.x.vals[2:end] .!= ref_traj.vi.metadata.x.vals[2:end]) == 0

        ancestor_particle = APS.get_new_trace(new_vi, selected_path.task, selected_path.taskinfo)
        vals = copy(ancestor_particle.vi.metadata.x.vals)

        score = Libtask.consume(ancestor_particle) # We need
        @test vals == ancestor_particle.vi.metadata.x.vals
        consume(pc)
        @test vals == ancestor_particle.vi.metadata.x.vals
        @test pc[1].vi.num_produce == ancestor_particle.vi.num_produce
        @test  sum(base_ref.vi.metadata.x.vals .!= ref_traj.vi.metadata.x.vals) == 0

        indx = [2,3]
        pold_2 = copy(pc[2])
        push!(indx,length(pc))
        resample!(pc, uf, indx, ref_traj, ancestor_particle)
        @test ancestor_particle.vi.metadata.x.vals[2:end] == ref_traj.vi.metadata.x.vals[2:end]

        ref_traj = ancestor_particle
        @test sum(ref_traj.vi.metadata.x.vals[2:end] .!= base_ref.vi.metadata.x.vals[2:end]) == 0
        @test pc[3] === ref_traj
        @test sum(pc[1].vi.metadata.x.vals .!= pold_2.vi.metadata.x.vals) == 0
        @test sum(pc[2].vi.x.flags["del"] .!= pc[1].vi.x.flags["del"]) == 0
        @test sum(pc[2].vi.x.flags["del"]) == 58
        @test sum(pc[3].vi.x.flags["del"]) == 0
        ancestor_index = 1
        # We are one step behind....
        selected_path = pc[ancestor_index]
        new_vi = uf.merge_traj!(copy(selected_path.vi), ref_traj.vi) # Merge trajectories
        @test sum(new_vi.metadata.x.vals[1:2] .!= selected_path.vi.metadata.x.vals[1:2]) == 0
        @test sum(new_vi.metadata.x.vals[3:end] .!= ref_traj.vi.metadata.x.vals[3:end]) == 0
        ancestor_particle = APS.get_new_trace(new_vi, selected_path.task, selected_path.taskinfo)



        vals = copy(ancestor_particle.vi.metadata.x.vals)
        score = Libtask.consume(ancestor_particle) # We need
        @test vals == ancestor_particle.vi.metadata.x.vals

        pc[1].vi.metadata.x.vals .== ancestor_particle.vi.metadata.x.vals
        ancestor_particle.vi.metadata.x.vals



        consume(pc)




        @test pc[1].vi.num_produce == ancestor_particle.vi.num_produce
        @test sum(base_ref.vi.metadata.x.vals[2:end] .!= ref_traj.vi.metadata.x.vals[2:end]) == 0

        indx = [2,3]
        pold_2 = copy(pc[2])
        push!(indx,length(pc))
        resample!(pc, uf, indx, ref_traj, ancestor_particle)
        ref_traj = ancestor_particle
        @test sum(ref_traj.vi.metadata.x.vals[3:end] .!= base_ref.vi.metadata.x.vals[3:end]) == 0
        ref_traj.vi.metadata.x.vals[3:end] .!= base_ref.vi.metadata.x.vals[3:end]
        @test pc[3] === ref_traj
        @test sum(pc[1].vi.metadata.x.vals .!= pold_2.vi.metadata.x.vals) == 0
        @test sum(pc[2].vi.x.flags["del"] .!= pc[1].vi.x.flags["del"]) == 0
        @test sum(pc[2].vi.x.flags["del"]) == 57
        @test sum(pc[3].vi.x.flags["del"] ) == 0

        ancestor_index = 1
        # We are one step behind....
        selected_path = pc[ancestor_index]
        new_vi = uf.merge_traj!(copy(selected_path.vi), ref_traj.vi) # Merge trajectories
        @test sum(new_vi.metadata.x.vals[1:3] .!= selected_path.vi.metadata.x.vals[1:3]) == 0
        @test sum(new_vi.metadata.x.vals[4:end] .!= ref_traj.vi.metadata.x.vals[4:end]) == 0
        ancestor_particle = APS.get_new_trace(new_vi, selected_path.task, selected_path.taskinfo)
        score = Libtask.consume(ancestor_particle) # We need


    end
end
