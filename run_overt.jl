using OVERTVerify
using LazySets
using Joseki, JSON, HTTP

### Create some endpoints

# This function takes two numbers x and y from the query string and returns x^y
# In this case they need to be identified by name and it should be called with
# something like 'http://localhost:8000/pow/?x=2&y=3'
function pow(req::HTTP.Request)
    j = HTTP.queryparams(HTTP.URI(req.target))
    has_all_required_keys(["x", "y"], j) || return error_responder(req, "You need to specify values for x and y!")
    # Try to parse the values as numbers.  If there's an error here the generic
    # error handler will deal with it.
    x = parse(Float32, j["x"])
    y = parse(Float32, j["y"])
    json_responder(req, x^y)
end

# This function takes two numbers n and k from a JSON-encoded request
# body and returns binomial(n, k)
function bin(req::HTTP.Request)
    j = try
        body_as_dict(req)
    catch err
        return error_responder(req, "I was expecting a json request body!")
    end
    has_all_required_keys(["n", "k"], j) || return error_responder(req, "You need to specify values for n and k!")
    json_responder(req, binomial(j["n"],j["k"]))
end

# This function takes two numbers n and k from a JSON-encoded request
# body and returns binomial(n, k)
function overtabc(req::HTTP.Request)
    j = try
        body_as_dict(req)
    catch err
        return error_responder(req, "I was expecting a json request body!")
    end
    has_all_required_keys(["input_set", "num_timesteps"], j) || return error_responder(req, "You need to specify other values!")

    ranges = setup_overt(j["input_set"], j["num_timesteps"])

    json_responder(req, ranges)
end

function setup_overt(input_set, num_timesteps)
    controller = "/home/mfe/code/OVERTVerify.jl/nnet_files/jmlr/single_pendulum_small_controller.nnet"

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


###############

using OVERT

pend_mass, pend_len, grav_const, friction = 0.5, 0.5, 1., 0.

function single_pend_dynamics(x::Array{T, 1} where {T <: Real},
                              u::Array{T, 1} where {T <: Real})
    dx1 = x[2]
    dx2 = grav_const/pend_len * sin(x[1]) + 1 / (pend_mass*pend_len^2) * (u[1] - friction * x[2])
    return [dx1, dx2]
end

single_pend_θ_doubledot = :($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)

function single_pend_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
                                    N_OVERT::Int,
                                    t_idx::Union{Int, Nothing}=nothing)
    if isnothing(t_idx)
        v1 = single_pend_θ_doubledot
    else
        v1 = "$(grav_const/pend_len) * sin(x1_$t_idx) + $(1/(pend_mass*pend_len^2)) * u1_$t_idx - $(friction/(pend_mass*pend_len^2)) * x2_$t_idx"
        v1 = Meta.parse(v1)
    end
    v1_oA = overapprox(v1, range_dict; N=N_OVERT)
    return v1_oA, [v1_oA.output]
end

function single_pend_update_rule(input_vars::Array{Symbol, 1},
                                 control_vars::Array{Symbol, 1},
                                 overt_output_vars::Array{Symbol, 1})
    ddth = overt_output_vars[1]
    integration_map = Dict(input_vars[1] => input_vars[2], input_vars[2] => ddth)
    return integration_map
end

single_pend_input_vars = [:x1, :x2]
single_pend_control_vars = [:u1]

SinglePendulumABC = OvertProblem(
    single_pend_dynamics,
    single_pend_dynamics_overt,
    single_pend_update_rule,
    single_pend_input_vars,
    single_pend_control_vars
)


##################

###############

using OVERT
using OVERT: add_overapproximate

function double_integrator_dynamics(x::Array{T, 1} where {T <: Real},
                              u::Array{T, 1} where {T <: Real})
    dx1 = x[1] + x[2] + 0.5*u[1]
    dx2 = x[2] + u[1]
    return [dx1, dx2]
end

tmp_tbd = :(x1 + x2 + 0.5*u1)
tmp_tbd2 = :(x2 + u1)

function double_integrator_dynamics_overt(range_dict::Dict{Symbol, Array{T, 1}} where {T <: Real},
                                    N_OVERT::Int,
                                    t_idx::Union{Int, Nothing}=nothing)
    if isnothing(t_idx)
        v1 = tmp_tbd
        v2 = tmp_tbd2
    else
        v1 = "x1_$t_idx + x2_$t_idx + 0.5*u1_$t_idx"
        v1 = Meta.parse(v1)
        v2 = "x2_$t_idx + u1_$t_idx"
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


##################

# input_set = Dict("low" => [0.5, 0.6], "high" => [0.7, 0.8])
# num_timesteps = 2
# concrete_sets = setup_overt(input_set, num_timesteps)
# print(concrete_sets)

# ### Create and run the server

# Make a router and add routes for our endpoints.
endpoints = [
    (pow, "GET", "/pow"),
    (bin, "POST", "/bin"),
    (overtabc, "POST", "/overt")
]
r = Joseki.router(endpoints)

# Fire up the server
HTTP.serve(r, "127.0.0.1", 8000; verbose=false)