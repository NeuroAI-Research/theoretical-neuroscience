import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax, random
from jax.nn import relu
from jax.random import PRNGKey, normal, uniform


def c7p2_firing_rate_models():
    tau_r = 0.01  # time constant: 10ms
    dt = 0.001  # simulation time step: 1ms
    tot_t = 0.1
    n_steps = int(tot_t / dt)
    n_in, n_out = 5, 4  # network size
    k1, k2, k3 = random.split(PRNGKey(42), 3)

    W = normal(k1, (n_out, n_in)) * 0.5  # feedforward weights
    M = normal(k2, (n_out, n_out)) * 0.5  # recurrent weights
    u = uniform(k3, (n_in,))

    def F(I):
        return relu(I - 0.1)

    def rate_dynamics(v, _):
        I = jnp.dot(W, u) + jnp.dot(M, v)
        dv_dt = (-v + F(I)) / tau_r
        v_next = v + dv_dt * dt
        return v_next, v_next

    v_init = jnp.zeros(n_out)
    xs = jnp.arange(n_steps)
    _, v_hist = lax.scan(rate_dynamics, v_init, xs)

    # ANN: dv_dt = 0, M = 0
    v_steady_M0 = F(jnp.dot(W, u))

    R, C = 2, 2
    for i in range(n_out):
        plt.subplot(R, C, i + 1)
        plt.plot(xs * dt, v_hist[:, i], ".", label="dynamic & M != 0")
        plt.axhline(v_steady_M0[i], label="steady & M = 0", c="red", linestyle="--")
        plt.title(f"neuron {i}")
        plt.legend()
    plt.tight_layout()
    plt.savefig("c7p2_firing_rate_models")


if __name__ == "__main__":
    c7p2_firing_rate_models()
