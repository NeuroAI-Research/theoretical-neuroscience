import jax.numpy as jnp
import jax.numpy.fft as F

from theoretical_neuroscience.plot import plot1
from theoretical_neuroscience.utils import frame_to_jax, read_video


def surprise(prob):
    return jnp.where(prob > 0, -jnp.log2(prob), 0)


def entropy(prob_dist):
    p = prob_dist
    return jnp.sum(p * surprise(p))


def cont_entropy(p_r, dr):
    return jnp.sum(dr * p_r * surprise(p_r))


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


# =======================


def image_fft(img: jnp.ndarray, filter="band_pass"):
    freq = F.fftshift(F.fft2(img))  # shift zero freq to center

    w, h = img.shape
    u, v = jnp.meshgrid(jnp.linspace(-1, 1, w), jnp.linspace(-1, 1, h))
    kappa = jnp.sqrt(u**2 + v**2)
    if filter == "band_pass":
        filter = kappa * jnp.exp(-(kappa**2) / 0.5)

    img2 = jnp.abs(F.ifft2(F.ifftshift(freq * filter)))
    return img2, filter


def c4p2_information_and_entropy_maximization():
    # optimal tuning curve
    s = jnp.linspace(-5, 5, 1000)
    p_s = 0.7 * jnp.exp(-((s - 1) ** 2) / 0.5) + 0.3 * jnp.exp(-((s + 2) ** 2) / 2)
    p_s /= jnp.trapezoid(p_s, s)  # normalize
    r_max = 100
    f_s_opt = r_max * jnp.cumsum(p_s) * (s[1] - s[0])

    # fft: band pass filtering
    for img in read_video("./data/cat_dance.mp4"):
        img = frame_to_jax(img)
        img2, filter = image_fft(img)
        break

    plots = {
        "p(s)": [s, p_s],
        "optimal f(s)": [s, f_s_opt],
        "img.img": img,
        "FFT_band_pass.img": filter,
        "new_img.img": img2,
    }
    plot1(plots, "c4p2_information_and_entropy_maximization")


if __name__ == "__main__":
    c4p2_information_and_entropy_maximization()
