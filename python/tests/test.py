import jax.numpy as jnp

from theoretical_neuroscience.plot import plot1


def surprise(prob):
    return jnp.where(prob > 0, -jnp.log2(prob), 0)


def entropy(prob_dist):
    p = prob_dist
    return jnp.sum(p * surprise(p))


def kl_divergence(prob_dist1, prob_dist2):
    p1, p2 = prob_dist1, prob_dist2
    mask = (p1 > 0) & (p2 > 0)
    p1, p2 = p1[mask], p2[mask]
    return jnp.sum(p1 * jnp.log2(p1 / p2))


def c4p1_entropy_and_mutual_information():
    prob = jnp.linspace(0, 1, 200)
    sur = surprise(prob)
    ent = entropy(prob / prob.sum())
    print(f"ent: {ent}")
    plots = {"surprise": [prob, sur]}
    plot1(plots, "c4p1_entropy_and_mutual_information")


if __name__ == "__main__":
    c4p1_entropy_and_mutual_information()
