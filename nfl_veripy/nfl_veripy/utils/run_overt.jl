
##################################
# Simple query of OVERT for closed-loop reachability
##################################

using OVERTVerify
using LazySets

function setup_overt(input_set, system, num_timesteps, controller, dt, nn_encoding)

    problem = systems_map[system]

    query = OvertQuery(
        problem,           # problem
        controller,        # network file
        Id(),              # last layer activation layer Id()=linear, or ReLU()=relu
        "MIP",             # query solver, "MIP" or "ReluPlex"
        num_timesteps,     # ntime
        dt,                # dt
        -1,                # N_overt
        )
    input_set_ = Hyperrectangle(low=input_set["low"], high=input_set["high"])
    concretization_intervals = [num_timesteps]

    concrete_state_sets, symbolic_state_sets, concrete_meas_sets, symbolic_meas_sets = symbolic_reachability_with_concretization(query, input_set_, concretization_intervals, nn_encoding=nn_encoding)

    # Convert into array form for python    
    num_states = length(input_set["low"])
    concrete_state_sets_range = zeros((2, num_states, num_timesteps))
    for t = 1:num_timesteps
        concrete_state_sets_range[1, :, t] = low(concrete_state_sets[1][t+1])
        concrete_state_sets_range[2, :, t] = high(concrete_state_sets[1][t+1])
    end

    return concrete_state_sets_range
end

##################################
# double integrator dynamics
##################################

using OVERT
using OVERT: add_overapproximate

function double_integrator_dynamics(x::Array{T, 1} where {T <: Real},
                              u::Array{T, 1} where {T <: Real})
    dx1 = x[2] + 0.5*u[1]
    dx2 = u[1]
    # dx1 = x[2] + 0.5*clamp(u[1], -1, 1)
    # dx2 = clamp(u[1], -1, 1)
    return [dx1, dx2]
end

tmp1_double_integrator = :(x2 + 0.5*u1)
tmp2_double_integrator = :(u1)
# tmp1_double_integrator = :(x2 + 0.5*clamp(u1, -1, 1))
# tmp2_double_integrator = :(clamp(u1, -1, 1))

function double_integrator_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
                                    N_OVERT::Int,
                                    t_idx::Union{Int, Nothing}=nothing)
    if isnothing(t_idx)
        v1 = tmp1_double_integrator
        v2 = tmp2_double_integrator
    else
        v1 = "x2_$t_idx + 0.5*u1_$t_idx"
        v2 = "u1_$t_idx"
        # v1 = "x2_$t_idx + 0.5*clamp(u1_$t_idx, -1, 1)"
        # v2 = "clamp(u1_$t_idx, -1, 1)"
        v1 = Meta.parse(v1)
        v2 = Meta.parse(v2)
    end
    v1_oA = overapprox(v1, range_dict; N=N_OVERT)
    v2_oA = overapprox(v2, range_dict; N=N_OVERT)
    oA_out = add_overapproximate([v1_oA, v2_oA])

    return oA_out, [v1_oA.output, v2_oA.output]
end

function double_integrator_update_rule(input_vars::Array{Symbol, 1},
                                 control_vars::Array{Symbol, 1},
                                 overt_output_vars::Array{Symbol, 1})
    integration_map = Dict(
        input_vars[1] => overt_output_vars[1],
        input_vars[2] => overt_output_vars[2]
    )
    return integration_map
end

double_integrator_input_vars = [:x1, :x2]
double_integrator_control_vars = [:u1]

DoubleIntegrator = OvertProblem(
    double_integrator_dynamics,
    double_integrator_dynamics_overt,
    double_integrator_update_rule,
    double_integrator_input_vars,
    double_integrator_control_vars
)

##################################
# quadrotor dynamics
##################################

g = 9.8

function quadrotor_dynamics(x::Array{T, 1} where {T <: Real},
                              u::Array{T, 1} where {T <: Real})
    dx1 = x[4]
    dx2 = x[5]
    dx3 = x[6]
    dx4 = g*u[1]
    dx5 = -g*u[2]
    dx6 = u[3] - g
    return [dx1, dx2, dx3, dx4, dx5, dx6]
end

tmp1_quadrotor = :(x4)
tmp2_quadrotor = :(x5)
tmp3_quadrotor = :(x6)
tmp4_quadrotor = :(g*u1)
tmp5_quadrotor = :(-g*u2)
tmp6_quadrotor = :(u3-g)

function quadrotor_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
                                    N_OVERT::Int,
                                    t_idx::Union{Int, Nothing}=nothing)
    if isnothing(t_idx)
        v1 = tmp1_quadrotor
        v2 = tmp2_quadrotor
        v3 = tmp3_quadrotor
        v4 = tmp4_quadrotor
        v5 = tmp5_quadrotor
        v6 = tmp6_quadrotor
    else
        v1 = "x4_$t_idx"
        v2 = "x5_$t_idx"
        v3 = "x6_$t_idx"
        v4 = "$(g)*u1_$t_idx"
        v5 = "-$(g)*u2_$t_idx"
        v6 = "u3_$t_idx - $(g)"
        v1 = Meta.parse(v1)
        v2 = Meta.parse(v2)
        v3 = Meta.parse(v3)
        v4 = Meta.parse(v4)
        v5 = Meta.parse(v5)
        v6 = Meta.parse(v6)
    end
    v1_oA = overapprox(v1, range_dict; N=N_OVERT)
    v2_oA = overapprox(v2, range_dict; N=N_OVERT)
    v3_oA = overapprox(v3, range_dict; N=N_OVERT)
    v4_oA = overapprox(v4, range_dict; N=N_OVERT)
    v5_oA = overapprox(v5, range_dict; N=N_OVERT)
    v6_oA = overapprox(v6, range_dict; N=N_OVERT)
    oA_out = add_overapproximate([v1_oA, v2_oA, v3_oA, v4_oA, v5_oA, v6_oA])

    return oA_out, [v1_oA.output, v2_oA.output, v3_oA.output, v4_oA.output, v5_oA.output, v6_oA.output]
end

function quadrotor_update_rule(input_vars::Array{Symbol, 1},
                                 control_vars::Array{Symbol, 1},
                                 overt_output_vars::Array{Symbol, 1})
    integration_map = Dict(
        input_vars[1] => overt_output_vars[1],
        input_vars[2] => overt_output_vars[2],
        input_vars[3] => overt_output_vars[3],
        input_vars[4] => overt_output_vars[4],
        input_vars[5] => overt_output_vars[5],
        input_vars[6] => overt_output_vars[6],
    )
    return integration_map
end

quadrotor_input_vars = [:x1, :x2, :x3, :x4, :x5, :x6]
quadrotor_control_vars = [:u1, :u2, :u3]

Quadrotor = OvertProblem(
    quadrotor_dynamics,
    quadrotor_dynamics_overt,
    quadrotor_update_rule,
    quadrotor_input_vars,
    quadrotor_control_vars
)

##################################
# unicycle dynamics
##################################

function unicycle_dynamics(x::Array{T, 1} where {T <: Real},
                              u::Array{T, 1} where {T <: Real})
    dx1 = u[1]*cos(x[3])
    dx2 = u[1]*sin(x[3])
    dx3 = u[2]
    return [dx1, dx2, dx3]
end

tmp1_unicycle = :(u1*cos(x3))
tmp2_unicycle = :(u1*sin(x3))
tmp3_unicycle = :(u2)

function unicycle_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
                                    N_OVERT::Int,
                                    t_idx::Union{Int, Nothing}=nothing)
    if isnothing(t_idx)
        v1 = tmp1_unicycle
        v2 = tmp2_unicycle
        v3 = tmp3_unicycle
    else
        v1 = "u1_$t_idx*cos(x3_$t_idx)"
        v2 = "u1_$t_idx*sin(x3_$t_idx)"
        v3 = "u2_$t_idx"

        v1 = Meta.parse(v1)
        v2 = Meta.parse(v2)
        v3 = Meta.parse(v3)
    end
    v1_oA = overapprox(v1, range_dict; N=N_OVERT)
    v2_oA = overapprox(v2, range_dict; N=N_OVERT)
    v3_oA = overapprox(v3, range_dict; N=N_OVERT)
    oA_out = add_overapproximate([v1_oA, v2_oA, v3_oA])

    return oA_out, [v1_oA.output, v2_oA.output, v3_oA.output]
end

function unicycle_update_rule(input_vars::Array{Symbol, 1},
                                 control_vars::Array{Symbol, 1},
                                 overt_output_vars::Array{Symbol, 1})
    integration_map = Dict(
        input_vars[1] => overt_output_vars[1],
        input_vars[2] => overt_output_vars[2],
        input_vars[3] => overt_output_vars[3]
    )
    return integration_map
end

unicycle_input_vars = [:x1, :x2, :x3]
unicycle_control_vars = [:u1, :u2]

Unicycle = OvertProblem(
    unicycle_dynamics,
    unicycle_dynamics_overt,
    unicycle_update_rule,
    unicycle_input_vars,
    unicycle_control_vars
)


##################################
# linearized unicycle dynamics
##################################

function linearized_unicycle_dynamics(x::Array{T, 1} where {T <: Real},
    u::Array{T, 1} where {T <: Real})
dx1 = u[1]
dx2 = u[2]
return [dx1, dx2]
end

tmp1_linearized_unicycle = :(u1)
tmp2_linearized_unicycle = :(u2)

function linearized_unicycle_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
          N_OVERT::Int,
          t_idx::Union{Int, Nothing}=nothing)
    if isnothing(t_idx)
        v1 = tmp1_linearized_unicycle
        v2 = tmp2_linearized_unicycle
    else
        v1 = "u1_$t_idx"
        v2 = "u2_$t_idx"


    v1 = Meta.parse(v1)
    v2 = Meta.parse(v2)
    end
    v1_oA = overapprox(v1, range_dict; N=N_OVERT)
    v2_oA = overapprox(v2, range_dict; N=N_OVERT)
    oA_out = add_overapproximate([v1_oA, v2_oA])

    return oA_out, [v1_oA.output, v2_oA.output]
end

function linearized_unicycle_update_rule(input_vars::Array{Symbol, 1},
       control_vars::Array{Symbol, 1},
       overt_output_vars::Array{Symbol, 1})
    integration_map = Dict(
    input_vars[1] => overt_output_vars[1],
    input_vars[2] => overt_output_vars[2]
    )
    return integration_map
end

linearized_unicycle_input_vars = [:x1, :x2]
linearized_unicycle_control_vars = [:u1, :u2]

LinearizedUnicycle = OvertProblem(
linearized_unicycle_dynamics,
linearized_unicycle_dynamics_overt,
linearized_unicycle_update_rule,
linearized_unicycle_input_vars,
linearized_unicycle_control_vars
)

systems_map = Dict(
    "DoubleIntegrator" => DoubleIntegrator,
    "Quadrotor" => Quadrotor,
    "Unicycle" => Unicycle,
    "GroundRobotSI" => LinearizedUnicycle
)


##################################
# call w/o http server
##################################

function simple_testcase()
    input_set = Dict("low" => [0.5, 0.6], "high" => [0.7, 0.8])
    system = "DoubleIntegrator"
    dt = 1.0
    num_timesteps = 2
    # controller = "/home/mfe/code/nn_robustness_analysis/nfl_veripy/models/nnet/tmp_model.nnet"
    controller = "/Users/mfe/code/OVERTVerify.jl/nnet_files/jmlr/single_pendulum_small_controller.nnet"
    # controller = "/home/mfe/code/OVERTVerify.jl/nnet_files/jmlr/acc_controller.nnet"

    # nn_encoding = "LP"
    nn_encoding = "MIP"

    concrete_sets = setup_overt(input_set, system, num_timesteps, controller, dt, nn_encoding)
    print(concrete_sets)
end

##################################
# enable calling julia over http server
##################################

### Create and run the server
using Joseki, JSON, HTTP

function overtabc(req::HTTP.Request)
    j = try
        body_as_dict(req)
    catch err
        return error_responder(req, "I was expecting a json request body!")
    end
    has_all_required_keys(["input_set", "num_timesteps", "controller", "system", "dt"], j) || return error_responder(req, "You need to specify other values!")

    # nn_encoding = "LP"
    nn_encoding = "MIP"

    ranges = setup_overt(j["input_set"], j["system"], j["num_timesteps"], j["controller"], j["dt"], nn_encoding)

    json_responder(req, ranges)
end

function start_server()
    # Make a router and add routes for our endpoints.
    endpoints = [
        (overtabc, "POST", "/overt")
    ]
    r = Joseki.router(endpoints)

    # Fire up the server
    HTTP.serve(r, "127.0.0.1", 8000; verbose=false)
end

# simple_testcase()

start_server()


