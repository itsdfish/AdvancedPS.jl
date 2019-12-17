
@model gdemo_d() = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return s, m
end

gdemo_default = gdemo_d()

function gdemo_d_apf()
    var = APSTCont.initialize()
    r = rand(InverseGamma(2, 3))
    vn = @varname s
    s = APSTCont.update_var!(var, vn, r)

    r =  rand(Normal(0, sqrt(s)))
    vn = @varname m
    m = APSTCont.update_var!(var, vn, r)

    logp = logpdf(Normal(m, sqrt(s)), 1.5)
    APSTCont.report_observation!(var,logp)

    logp = logpdf(Normal(m, sqrt(s)), 2.0)
    APSTCont.report_observation!(var,logp)
end


aps_gdemo_default = gdemo_d_apf

@model large_demo(y) = begin
    x = TArray{Float64}(undef,11)
    x[1] ~ Normal()
    for i = 2:11
        x[i] ~ Normal(0.1*x[i-1]-0.1,0.5)
        y[i-1] ~ Normal(x[i],0.3)
    end
end

function large_demo_apf(y)
    var = APSTCont.initialize()
    x = Turing.TArray{Float64}(undef,11)
    vn = @varname x[1]
    x[1] = APSTCont.update_var!(var, vn, rand(Normal()))
    # there is nothing to report since we are not using proposal sampling
    logp = logpdf(Normal(), x[1])
    APSTCont.report_transition!(var,logp,logp)
    for i = 2:11
        # Sampling
        r =  rand( Normal(0.1*x[i-1]-0.1,0.5))
        vn = @varname x[i]
        x[i] = APSTCont.update_var!(var, vn, r)
        logγ = logpdf( Normal(0.1*x[i-1]-0.1,0.5),x[i]) #γ(x_t|x_t-1)
        logp = logγ             # p(x_t|x_t-1)
        APSTCont.report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(Normal(x[i], 0.3), y[i-1])
        var = APSTCont.report_observation!(var,logpy)
    end
end

## Here, we even have a proposal distributin
function large_demo_apf_proposal(y)
    var = APSTCont.initialize()
    x = Turing.TArray{Float64}(undef,11)
    vn = @varname x[1]
    x[1] = APSTCont.update_var!(var, vn, rand(Normal()))
    logp = logpdf(Normal(), x[1])

    # there is nothing to report since we are not using proposal sampling
    APSTCont.report_transition!(var,logp,logp)
    for i = 2:11
        # Sampling
        r =  rand(Normal(0.1*x[i-1],0.5))
        vn = @varname x[i]
        x[i] = APSTCont.update_var!(var, vn, r)
        logγ = logpdf(Normal(0.1*x[i-1],0.5),x[i]) #γ(x_t|x_t-1)
        logp = logpdf( Normal(0.1*x[i-1]-0.1,0.5),x[i])             # p(x_t|x_t-1)
        APSTCont.report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(Normal(x[i], 0.3), y[i-1])
        var = APSTCont.report_observation!(var,logpy)
    end
end

function large_demo_apf_c(y)
    var = CustomCont.initialize()
    x = Turing.TArray{Float64}(undef,11)
    x[1:1] = CustomCont.update_var!(var, 1, rand(Normal(),1))
    # there is nothing to report since we are not using proposal sampling
    logp = logpdf(Normal(), x[1])

    CustomCont.report_transition!(var,logp,logp)
    for i = 2:11
        # Sampling
        r =  rand( Normal(0.1*x[i-1]-0.1,0.5),1)
        x[i:i] = CustomCont.update_var!(var, i, r)
        logγ = logpdf( Normal(0.1*x[i-1]-0.1,0.5),x[i]) #γ(x_t|x_t-1)
        logp = logγ             # p(x_t|x_t-1)
        CustomCont.report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(Normal(x[i], 0.3), y[i-1])
        var = CustomCont.report_observation!(var,logpy)
    end
end

## Here, we even have a proposal distributin
function large_demo_apf_proposal_c(y)
    var = CustomCont.initialize()
    x = Turing.TArray{Float64}(undef,11)
    x[1:1] = CustomCont.update_var!(var, 1, rand(Normal(),1))
    logp = logpdf(Normal(), x[1])

    # there is nothing to report since we are not using proposal sampling
    CustomCont.report_transition!(var,logp,logp)
    for i = 2:11
        # Sampling
        r =  rand(Normal(0.1*x[i-1],0.5),1)
        x[i:i] = CustomCont.update_var!(var, i, r)
        logγ = logpdf(Normal(0.1*x[i-1],0.5),x[i]) #γ(x_t|x_t-1)
        logp = logpdf( Normal(0.1*x[i-1]-0.1,0.5),x[i])             # p(x_t|x_t-1)
        CustomCont.report_transition!(var,logp,logγ)
        #Proposal and Resampling
        logpy = logpdf(Normal(x[i], 0.3), y[i-1])
        var = CustomCont.report_observation!(var,logpy)
    end
end
