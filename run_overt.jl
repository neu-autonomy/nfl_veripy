
##################################
# Simple query of OVERT for closed-loop reachability
##################################

using OVERTVerify
using LazySets

function setup_overt(input_set, num_timesteps, controller)

    query = OvertQuery(
        DoubleIntegrator,  # problem
        controller,        # network file
        Id(),              # last layer activation layer Id()=linear, or ReLU()=relu
        "MIP",             # query solver, "MIP" or "ReluPlex"
        num_timesteps,     # ntime
        1.0,               # dt
        -1,                # N_overt
        )
    input_set_ = Hyperrectangle(low=input_set["low"], high=input_set["high"])
    concretization_intervals = [num_timesteps]
    concrete_state_sets, symbolic_state_sets, concrete_meas_sets, symbolic_meas_sets = symbolic_reachability_with_concretization(query, input_set_, concretization_intervals)

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

tmp_tbd = :(x2 + 0.5*u1)
tmp_tbd2 = :(u1)
# tmp_tbd = :(x2 + 0.5*clamp(u1, -1, 1))
# tmp_tbd2 = :(clamp(u1, -1, 1))

function double_integrator_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
                                    N_OVERT::Int,
                                    t_idx::Union{Int, Nothing}=nothing)
    if isnothing(t_idx)
        v1 = tmp_tbd
        v2 = tmp_tbd2
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
# call w/o http server
##################################

# input_set = Dict("low" => [0.5, 0.6], "high" => [0.7, 0.8])
# num_timesteps = 2
# controller = "/home/mfe/code/nn_robustness_analysis/nn_closed_loop/models/nnet/tmp_model.nnet"
# # controller = "/home/mfe/code/OVERTVerify.jl/nnet_files/jmlr/single_pendulum_small_controller.nnet"
# # controller = "/home/mfe/code/OVERTVerify.jl/nnet_files/jmlr/acc_controller.nnet"
# concrete_sets = setup_overt(input_set, num_timesteps, controller)
# print(concrete_sets)

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
    has_all_required_keys(["input_set", "num_timesteps", "controller"], j) || return error_responder(req, "You need to specify other values!")

    ranges = setup_overt(j["input_set"], j["num_timesteps"], j["controller"])

    json_responder(req, ranges)
end

# Make a router and add routes for our endpoints.
endpoints = [
    (overtabc, "POST", "/overt")
]
r = Joseki.router(endpoints)

# Fire up the server
HTTP.serve(r, "127.0.0.1", 8000; verbose=false)