
# 1 Neural Encoding I

## 1.2 Spike Trains and Firing Rates

- **Neural Response Function**: a sequence of $n$ spikes (Dirac delta) at times $t_i$

$$ \rho(t) = \sum_{i=1}^{n} \delta(t - t_{i}) $$

- **Spike-Count Rate:** A single-trial average over time $T$

$$ r = \frac{n}{T} = \frac{1}{T} \int_{0}^{T} d\tau \rho(\tau) $$

- **Firing Rate:** The average number of spikes across **multiple trials** in a short interval $\Delta t$

$$ r(t) = \frac{1}{\Delta t} \int_{t}^{t+\Delta t} d\tau \langle \rho(\tau) \rangle $$

- Since we have limited data, we approximate $r(t)$ by convolving the spike train with a **window function** (kernel) $w$
    - **Rectangular:** Standard binning; placement can be arbitrary
    - **Gaussian:** Smooths the rate; uses future and past spikes
    - **Causal (Alpha Function):** Physically realistic; depends only on spikes that have already occurred

$$ r_{approx}(t) = \int_{-\infty}^{\infty} d\tau w(\tau) \rho(t-\tau) \\[5pt]
w(\tau) = [\alpha^{2} \tau \exp(-\alpha\tau)]_{+} $$

---

- The **Response Tuning Curve** $f(s)$ is the average firing rate $\langle r \rangle$ expressed as a function of a stimulus attribute $s$

| Geometry | Equation | Context |
| --- | --- | --- |
| **Gaussian** | $f(s) = r_{max} \exp \left[ -\frac{1}{2} \left( \frac{s - s_{max}}{\sigma_f} \right)^2 \right]$ | Orientation in V1 |
| **Cosine** | $f(s) = [r_0 + (r_{max} - r_0) \cos(s - s_{max})]_+$ | Reaching angle in M1 |
| **Sigmoid** | $f(s) = \frac{r_{max}}{1 + \exp((s_{1/2} - s)/\Delta_s)}$ | Retinal disparity |
