



@testset "Custom_Container.jl" begin
    @apf_testset "copy container" begin
        n = 10
        tcontainer =  Container(zeros(1, n),Vector{Bool}(falses(n)),Vector{Int}(zeros(n)),0)
        cp = copy(tcontainer)
        cp.x[1,1] = 10
        cp.marked[4] = true
        cp.produced_at[4] = 5

        @test tcontainer.x[1,1] == 0.0
        @test tcontainer.marked[4]  == false
        @test tcontainer.produced_at[4]      == 0
        @test typeof(tcontainer) === typeof(cp)
    end

    @apf_testset "merge_traj" begin
        n = Ref(0)
        alg = AdvancedPS.PGASAlgorithm(5)
        uf = AdvancedPS.PGASUtilityFunctions(CustomCont.set_retained_vns_del_by_spl!,  CustomCont.tonamedtuple, CustomCont.merge_traj!)
        tcontainer =  Container(zeros(1, 60),Vector{Bool}(falses(60)),Vector{Int}(zeros(60)),0)

        dist = Normal(0, 1)

        function fpc()
            var = CustomCont.initialize()
            i = 0
            for k = 1:30
                r = rand(dist)
                i += 1
                CustomCont.update_var!(var, i, [r])
                produce(0)
                var = current_trace()
                r = rand(dist)
                i += 1
                CustomCont.update_var!(var, i, [r])
                produce(1)
                var = current_trace()
            end
            produce(Val(:done))
        end


        model = PFModel(fpc, NamedTuple())
        particles = [AdvancedPS.get_new_trace(tcontainer, model.task, APS.PGASTaskInfo(0.0, 0.0, 0.0)) for _ in 1:1]
        pc = ParticleContainer(particles)
        while consume(pc[1]) != Val(:done) end

        ref_traj = APS.forkr(pc[1])
        model = PFModel(fpc, NamedTuple())
        particles = [AdvancedPS.get_new_trace(tcontainer, model.task, APS.PGASTaskInfo(0.0, 0.0, 0.0)) for _ in 1:2]

        pc = ParticleContainer(particles)
        pc = push!(pc, ref_traj)

        base_ref = copy(ref_traj)

        consume(pc) == log(1)

        @test sum(base_ref.vi.x .!= ref_traj.vi.x) == 0

        indx = [2,3]
        pold_2 = copy(pc[2])
        push!(indx,length(pc))
        resample!(pc, uf, indx, ref_traj, nothing)
        @test pc[3] === ref_traj
        @test sum(pc[1].vi.x .!= pold_2.vi.x) == 0
        # Tests set_retained_vns_del_by_spl
        @test sum(pc[2].vi.marked .!= pc[1].vi.marked) ==0
        @test sum(pc[2].vi.marked) == 1
        @test sum(pc[3].vi.marked .== false) == 0


        ancestor_index = 1
        # We are one step behind....
        selected_path = pc[ancestor_index]
        new_vi = uf.merge_traj!(copy(selected_path.vi), ref_traj.vi) # Merge trajectories
        @test  sum(new_vi.x[:, 1] .!= selected_path.vi.x[:, 1]) == 0
        @test  sum(new_vi.x[:, 2:end] .!= ref_traj.vi.x[:, 2:end]) == 0
        ancestor_particle = APS.get_new_trace(new_vi, selected_path.task, selected_path.taskinfo)
        score = Libtask.consume(ancestor_particle) # We need
        consume(pc)
        @test pc[1].vi.num_produce == ancestor_particle.vi.num_produce
        @test  sum(base_ref.vi.x .!= ref_traj.vi.x) == 0

        indx = [2,3]
        pold_2 = copy(pc[2])
        push!(indx,length(pc))
        resample!(pc, uf, indx, ref_traj, ancestor_particle)
        ref_traj = ancestor_particle
        @test sum(ref_traj.vi.x[:, 2:end] .!= base_ref.vi.x[:, 2:end]) == 0
        @test pc[3] === ref_traj
        @test sum(pc[1].vi.x .!= pold_2.vi.x) == 0
        @test sum(pc[2].vi.marked .!= pc[1].vi.marked) == 0
        @test sum(pc[2].vi.marked) == 2
        @test sum(pc[3].vi.marked .== false) == 0

        ancestor_index = 1
        # We are one step behind....
        selected_path = pc[ancestor_index]
        new_vi = uf.merge_traj!(copy(selected_path.vi), ref_traj.vi) # Merge trajectories
        @test sum(new_vi.x[:, 1:2] .!= selected_path.vi.x[:, 1:2]) == 0
        @test sum(new_vi.x[:, 3:end] .!= ref_traj.vi.x[:, 3:end]) == 0
        ancestor_particle = APS.get_new_trace(new_vi, selected_path.task, selected_path.taskinfo)
        score = Libtask.consume(ancestor_particle) # We need
        consume(pc)
        @test pc[1].vi.num_produce == ancestor_particle.vi.num_produce
        @test sum(base_ref.vi.x[:, 2:end] .!= ref_traj.vi.x[:, 2:end]) == 0

        indx = [2,3]
        pold_2 = copy(pc[2])
        push!(indx,length(pc))
        resample!(pc, uf, indx, ref_traj, ancestor_particle)
        ref_traj = ancestor_particle
        @test sum(ref_traj.vi.x[:, 3:end] .!= base_ref.vi.x[:, 3:end]) == 0
        @test pc[3] === ref_traj
        @test sum(pc[1].vi.x .!= pold_2.vi.x) == 0
        @test sum(pc[2].vi.marked .!= pc[1].vi.marked) == 0
        @test sum(pc[2].vi.marked) == 3
        @test sum(pc[3].vi.marked .== false) == 0

        ancestor_index = 1
        # We are one step behind....
        selected_path = pc[ancestor_index]
        new_vi = uf.merge_traj!(copy(selected_path.vi), ref_traj.vi) # Merge trajectories
        @test sum(new_vi.x[:, 1:3] .!= selected_path.vi.x[:, 1:3]) == 0
        @test sum(new_vi.x[:, 4:end] .!= ref_traj.vi.x[:, 4:end]) == 0
        ancestor_particle = APS.get_new_trace(new_vi, selected_path.task, selected_path.taskinfo)
        score = Libtask.consume(ancestor_particle) # We need
    end
end
