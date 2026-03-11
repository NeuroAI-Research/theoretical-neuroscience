from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from theoretical_neuroscience.plot import plot1


def gaussian_tuning_curve(s, amp=50, mu=0, std=10):
    return amp * jnp.exp(-0.5 * ((s - mu) / std) ** 2)


def causal_kernel(alpha, t):
    kernel = (alpha**2) * t * jnp.exp(-alpha * t)
    return kernel / jnp.sum(kernel)


@partial(jax.vmap, in_axes=(0, None, None))
def poisson_spikes(key, rates: jnp.ndarray, dt: float):
    probs = rates * dt
    spikes = random.uniform(key, rates.shape) < probs
    return spikes.astype(jnp.float32)


@partial(jax.vmap, in_axes=(0, None))
def spikes_to_rates(spikes, kernel):
    conv = jnp.convolve(spikes, kernel, mode="full")
    return conv[: len(spikes)]


def test_1p2():
    key = random.PRNGKey(42)
    dt, duration = 1e-3, 1
    t = jnp.arange(0, duration, dt)
    alpha = 100
    n_trials = 100

    stimulus_path = jnp.linspace(-20, 20, len(t))
    true_rates = gaussian_tuning_curve(stimulus_path)
    kernel = causal_kernel(alpha, t=jnp.arange(0, 0.1, dt))

    keys = random.split(key, n_trials)
    spikes = poisson_spikes(keys, true_rates, dt)
    rates = spikes_to_rates(spikes, kernel) / dt  # scale by 1/dt to get Hz

    plots = {
        "true_rates": true_rates,
        "kernel": kernel,
        "spikes[0]": spikes[0],
        "rates.mean(0)": rates.mean(0),
    }
    plot1(plots, "temp")


if __name__ == "__main__":
    test_1p2()
