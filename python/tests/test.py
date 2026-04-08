import jax.numpy as jnp
from jax import jit, lax

from theoretical_neuroscience.plot import plot1


def c9p2_classical_conditioning():
    n_trials = 600
    T = 300
    t_stimulus = 100
    t_reward = 200
    lr = 0.2
    max_lag = T - t_stimulus

    @jit
    def td_learning_step(carry, _):
        w, u, r = carry
        # The Prediction (Expected Future Reward):
        # $ v(t) = \sum_{\tau=0} w(\tau) u(t-\tau) $
        # special case: pulse stimulus, u = 1 when t = t_stimulus+\tau >= t_stimulus
        v = jnp.zeros(T)
        v = v.at[t_stimulus:].set(w)

        # The TD Error (Dopamine Signal):
        # $ \delta(t) = r(t) + v(t+1) - v(t) $
        v_next = jnp.roll(v, -1).at[-1].set(0.0)
        delta = r + v_next - v

        # The Weight Update (Synaptic Change):
        # $ w(\tau) \rightarrow w(\tau) + \epsilon \delta(t) u(t-\tau) $
        w_new = w + lr * delta[t_stimulus : t_stimulus + max_lag]

        return (w_new, u, r), (v, delta)

    w_init = jnp.zeros(max_lag) + 1e-5
    u = jnp.zeros(T).at[t_stimulus].set(1.0)
    r = jnp.zeros(T).at[t_reward].set(1.0)
    (w, _, _), (v, delta) = lax.scan(
        td_learning_step, (w_init, u, r), jnp.arange(n_trials)
    )

    trials = [0, 300, 599]
    plots = {}
    for i in trials:
        plots[f"v (t_rew={t_reward}, trial={i})"] = v[i]
        plots[f"delta (t_rew={t_reward}, trial={i})"] = delta[i]

    # t_reward = 150
    # r2 = jnp.zeros(T).at[t_reward].set(1.0)
    # _, (v, delta) = td_learning_step((w, u, r2), None)
    # plots[f"v (t_rew={t_reward})"] = v
    # plots[f"delta (t_rew={t_reward})"] = delta

    plot1(plots, "c9p2_classical_conditioning")


if __name__ == "__main__":
    c9p2_classical_conditioning()
