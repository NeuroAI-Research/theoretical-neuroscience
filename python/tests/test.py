import cv2
import jax.numpy as jnp
from jax import lax, random
from jax.nn import relu, sigmoid
from jax.random import PRNGKey

from theoretical_neuroscience.m1_neural_encoding import poisson_spikes
from theoretical_neuroscience.plot import plot1
from theoretical_neuroscience.utils import postfix


def predict_firing_rate(stimulus, kernel, r0, act="relu"):
    # L: linear filter output
    L = jnp.convolve(stimulus, kernel, mode="full")[: len(stimulus)]
    if act == "relu":
        F_L = relu(L)
    elif act == "sigmoid":
        r_max, g1, L_half = 100, 2, 3
        F_L = r_max * sigmoid(g1 * (L - L_half))
    return r0 + F_L


def demo_predict_firing_rate():
    key1 = PRNGKey(42)
    key2 = random.split(PRNGKey(42), 3)
    dt = 1e-3
    r0 = 100

    stimulus = random.normal(key1, (500,))
    kernel = jnp.exp(-jnp.linspace(0, 10, 50))

    r = predict_firing_rate(stimulus, kernel, r0)
    spikes = poisson_spikes(key2, r, dt)

    plots = {
        "stimulus": stimulus,
        "kernel": kernel,
        "firing rate": r,
        "spikes[0]": spikes[0],
    }
    plot1(plots, "3_demo_predict_firing_rate")


# =================================


def frame_to_jax(frame, c=cv2.COLOR_BGR2GRAY):
    return jnp.array(cv2.cvtColor(frame, c), dtype=jnp.float32)


def conv_2d(stimulus: jnp.ndarray, kernel_2d: jnp.ndarray):
    s = stimulus[None, None, :, :]
    k = kernel_2d[None, None, :, :]
    s2 = lax.conv_general_dilated(s, k, window_strides=(1, 1), padding="SAME")
    return s2.squeeze()


def retinal_step(state, frame, kernel_2d, debug=True, r0=5):
    key = state

    # 1. photoreceptor: normalize the current frame relative to its own mean
    # bio: rapid adaption to local light levels
    avg_lum = jnp.mean(frame)
    s = (frame - avg_lum) / (avg_lum + 1e-5)

    # 2. bipolar cells: center-surround filtering (linear filter)
    # math: L = ∫ dτ D s (2D spatial convolution)
    graded_potential = conv_2d(s, kernel_2d)

    # 3. ganglion cells: static nonlinearity & Poisson spiking
    # math: r_est = r0 + F(L)
    firing_rate = relu(r0 + graded_potential)
    this_key, next_key = random.split(key)
    spikes = random.poisson(this_key, firing_rate)

    if debug:
        plots = {
            "frame": frame,
            "kernel_2d": kernel_2d,
            "stimulus": s,
            "graded_potential": graded_potential,
            "firing_rate": firing_rate,
            "spikes": spikes,
        }
        plot1(postfix(plots, ".img"), "4_retinal_step")
    return next_key, firing_rate, spikes


def test_vision():
    path = "./data/eye.mp4"
    vid = cv2.VideoCapture(path)
    key = PRNGKey(42)
    kernel_2d = jnp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=jnp.float32)

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = frame_to_jax(frame)
        retinal_step(key, frame, kernel_2d)
        return
    vid.release()


if __name__ == "__main__":
    test_vision()
