"""Functions to initialize a simple NN control policy in jax."""
import functools

import jax
import jax.numpy as jnp


def random_layer_params(m: int, n: int, key: int, scale: int = 1e0):
    """Randomly initialize weights and biases for dense NN layer."""
    w_key, b_key = jax.random.split(key)
    return scale * jax.random.normal(w_key, (m, n)), scale * jax.random.normal(
        b_key, (n,)
    )


def init_network_params(sizes, key: int, scale: int = 1e0):
    """Initialize all layers for fully-connected NN with sizes "sizes"."""
    keys = jax.random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k, scale=scale)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def predict_mlp(params, u_limits, inputs):
    return predict_mlp_with_relu_control_limits(params, u_limits, inputs)
    # return predict_mlp_with_clip_control_limits(params, u_limits, inputs)


def predict_mlp_with_clip_control_limits(params, u_limits, inputs):
    return jnp.clip(
        predict_mlp_unclipped(params, inputs),
        jnp.array([u_limits[:, 0]]),
        jnp.array([u_limits[:, 1]]),
    )


def predict_mlp_with_sigmoid_control_limits(params, u_limits, inputs):
    return (u_limits[:, 1] - u_limits[:, 0]) * jax.nn.sigmoid(
        predict_mlp_unclipped(params, inputs)
    ) + u_limits[:, 0]


def predict_mlp_with_relu_control_limits(params, u_limits, inputs):
    ut = predict_mlp_unclipped(params, inputs)
    ut = -jnp.maximum(ut - u_limits[:, 0], 0) + u_limits[:, 1] - u_limits[:, 0]
    ut = -jnp.maximum(ut, 0) + u_limits[:, 1]
    return ut


def predict_mlp_unclipped(params, inputs):
    for W, b in params[:-1]:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.maximum(outputs, 0)
    W, b = params[-1]
    return jnp.dot(inputs, W) + b


def predict_next_state(params, dynamics, xt):
    ut = predict_mlp(params, dynamics.u_limits, xt)
    xt1 = dynamics.dynamics_step_jnp(xt, ut)
    return xt1


def predict_future_states(params, dynamics, num_timesteps, xt):
    xts = [xt]
    for _ in range(num_timesteps):
        ut = predict_mlp(params, dynamics.u_limits, xts[-1])
        xts.append(dynamics.dynamics_step_jnp(xts[-1], ut))
    return xts


def setup_nn(u_limits, num_states=2, num_control_inputs=1, scale=1e0):
    layer_sizes = [num_states, 5, 5, num_control_inputs]
    params = init_network_params(
        layer_sizes, jax.random.PRNGKey(0), scale=scale
    )
    logits_fn = functools.partial(predict_mlp, params, u_limits)
    batched_predict = jax.vmap(predict_mlp, in_axes=(None, 0))
    return params, batched_predict, logits_fn
