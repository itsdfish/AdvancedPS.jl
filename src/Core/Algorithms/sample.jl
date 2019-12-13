

## This is all we need for Turing!

function sample!(pc::PC, alg::ALG, uf::AbstractSMCUtilitFunctions, ref_traj::T) where {
    PC <:ParticleContainer,
    ALG <:Union{SMCAlgorithm, PGAlgorithm},
    T <:Union{Trace,Nothing}
}
    n = length(pc.vals)
    while consume(pc) != Val{:done}
        ess = effectiveSampleSize(pc)
        if ref_traj !== nothing || ess <= alg.resampler_threshold * length(pc)
            # compute weights
            Ws = weights(pc)

            # check that weights are not NaN
            @assert !any(isnan, Ws)
            # sample ancestor indices
            # Ancestor trajectory is not sampled
            ref_traj !== nothing ? nresamples = n-1 : nresamples = n
            indx = alg.resampler(Ws, nresamples)
            # We add ancestor trajectory to the path.
            # For ancestor sampling, we would change n at this point.
            ref_traj !== nothing ? push!(indx,n) : nothing
            resample!(pc, uf, indx, ref_traj)
        end
    end
end



function sample!(pc::PC, alg::ALG, uf::AbstractSMCUtilitFunctions, ref_traj::T) where {
    PC <:ParticleContainer,
    ALG <:Union{PGASAlgorithm},
    T <:Union{Trace,Nothing}
}
    # Before starting, we need to copute the ancestor weights.
    # Note that there is a inconsistency with the original paper
    # http://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf
    # Lindsten already samples x1. This is not possible in
    # in this situation because we do not have access to the information
    # which variables belong to x1 and which to x0!

    # The procedure works as follows. The ancestor weights are only dependent on
    # the states x_{0:t-1}. We make us of this by computing the ancestor indices for
    # the next state.
    ancestor_index = length(pc)
    n = length(pc)
    first_round = true
    while consume(pc) != Val{:done}
        # compute weights
        Ws = weights(pc)
        # We need them for ancestor sampling...
        logws = copy(pc.logWs)
        logpseq = [pc[i].taskinfo.logpseq for i in 1:n ]
        # check that weights are not NaN
        @assert !any(isnan, Ws)
        # sample ancestor indices
        # Ancestor trajectory is not sampled
        nresamples = n-1
        indx = resampler(Ws, nresamples)
        num_total_consume = typemax(Int64)
        # Now the ancestor sampling starts. We do not use ancestor sampling in the
        # first step. This is due to the reason before. In addition, we do not need
        # to compute the ancestor index for the last step, because we are always
        #computing the ancestor index one step ahead.
        if ref_traj !== nothing
            push!(indx,n)
            resample!(pc, indx,ref_particle)
            # In this case, we do have access to the joint_logp ! Therefore:
            if typeof(pc[1].taskinfo.ancestor_weight) !== Nothing

                for (i,t) in enumerate(pc.vals)
                    logws[i] += t.taskinfo.ancestor_weight - logpseq[i]  # The ancestor weights w_ancstor = w_i *p(x_{0:t-1},x_{t:T})/p(x_{0:t-1})
                end

            else
                if pc.n_consume <= num_total_consume-1 #We do not need to sample the last one...
                    # The idea is rather simple, we extend the vs and let them run trough...
                    pc_ancestor = ParticleContainer{typeof(pc[1].vi),typeof(pc[1].taskinfo)}()
                    pc_ancestor.n_consume = pc.n_consume
                    for i = 1:n
                        new_vi = uf.merge_traj!(copy(pc[i].vi),ref_particle.vi)
                        new_particle = get_new_trace(new_vi, pc[i].task, pc[i].taskinfo)
                        push!(pc_ancestor,new_particle)
                    end

                    while consume(pc_ancestor) != Val{:done} end # No resampling, we just want to get log p(x_{0:t-1},x'_{t,T})
                    for i in 1:n
                        logws[i] += pc_ancestor[i].taskinfo.logpseq -logpseq[i]  # The ancestor weights w_ancstor = w_i *p(x_{0:t-1},x_{t:T})/p(x_{0:t-1})
                    end
                    num_total_consume = pc_ancestor.n_consume
                end

                ancestor_index = randcat(softmax!(logws))
                # We are one step behind....
                selected_path = pc[ancestor_index]
                new_vi = uf.merge_traj(copy(selected_path.vi), ref_particle.vi, pc.n_consume +1)
                ancestor_particle = get_new_trace(new_vi, selected_path.task, selected_path.taskinfo)
            end
        end
        # Reactivate the tasks, this is only important for synchorinty.
    end


end
