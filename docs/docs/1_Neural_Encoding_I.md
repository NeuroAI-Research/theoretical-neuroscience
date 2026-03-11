
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




## 1.3 What Makes a Neuron Fire?

- Traditionally, neuroscience looks at how a **stimulus** causes a **response** (Encoding). This chapter reverses that question: given that a neuron just fired a spike, what did the world look like just a moment ago? This helps us identify the specific features that a neuron is "tuned" to detect.

- **Dynamic Range & Adaptation**: Neurons face a massive range of inputs (e.g., from a single photon to millions). To handle this, they respond to **changes** rather than steady states

- **Weber’s Law**: The "just noticeable difference" ($\Delta s$) is proportional to the stimulus intensity ($s$): 

$$ \Delta s/s = \text{constant} $$
 
- **Fechner’s Law**: Perceived intensity follows the logarithm of the actual intensity

$$ p \propto \log(s) $$ 

- **Simplification**: We often define $s(t)$ so its time-average is $0$, focusing only on the "fluctuations" around the mean

- **Spike-Triggered Average (STA)** is the "average profile" of the stimulus at time $\tau$ before a spike
    - **$t_i$**: The exact time of the $i$-th spike
    - **$\tau$**: The look-back time (delay)

$$ C(\tau) = \left\langle \frac{1}{n} \sum_{i=1}^{n} s(t_i - \tau) \right\rangle \\[5pt]
= \frac{1}{\langle n \rangle} \int_{0}^{T} dt \, r(t) s(t-\tau)
= \frac{1}{\langle r \rangle} Q_{rs}(-\tau)
$$

-  **Correlation Function Definition**:

$$ Q_{rs}(\tau) := \frac{1}{T} \int_{0}^{T} dt ~ r(t) s(t+\tau) $$

- To find out what a neuron truly likes, we need to test it with everything at once. **White noise** is a stimulus where the value at one time is completely uncorrelated with any other time.
    - **The Advantage**: It has a "flat power spectrum," meaning it tests the neuron at all frequencies with equal weight.
    - **Autocorrelation** $ Q_{ss}(\tau) = \sigma_s^2 \delta(\tau) $  : For white noise, the stimulus only "matches" itself at the exact same moment 

- **Multi-Spike Triggers**: Do "Bursts" mean something?
    - Sometimes a single spike doesn't tell the whole story. We can calculate the average stimulus triggered by a **pair** of spikes separated by time $\Delta t$
    - **Independent Spikes**: If the spikes are far apart (e.g., 10 ms), the two-spike average looks like two single-spike averages added together
    - **Synergy**: If they are very close (e.g., 5 ms), the average stimulus is different from the sum of two spikes, meaning the "burst" signals something unique that a single spike cannot






## 1.4 Spike-Train Statistics

- The Poisson Process Model
    - The fundamental assumption of a Poisson process is that each spike is generated independently of all others. This means the probability of a spike occurring at any given time depends only on the firing rate, $r(t)$, and not on the history of previous spikes

- To see if a natural phenomenon is Poisson, ask: **"Does the timing of the last event change the probability of the next one?"**
    - **No:** It’s Poisson
    - **Yes, it makes it less likely:** It’s a Renewal Process (more regular)
    - **Yes, it makes it more likely:** It’s a Hawkes/Clustered Process (more bursty)

- **Spike Count Probability (Poisson Distribution):** The probability of observing exactly $n$ spikes in an interval of duration $T$ for a constant rate $r$ is:

$$ P_T[n] = \frac{(rT)^n}{n!} \exp(-rT) $$

- **Mean and Variance:** For a Poisson process, the mean number of spikes $\langle n \rangle$ and the variance $\sigma_n^2$ are equal:

$$ \langle n \rangle = \sigma_n^2 = rT $$

- **Inter-Spike Interval (ISI) Distribution:** The time interval $\tau$ between successive spikes follows an exponential distribution:

$$ p[\tau] = r \exp(-r\tau) $$

- **Coefficient of Variation ($C_V$):** The ratio of the standard deviation of the ISIs to their mean. For a Poisson process, $C_V = 1$

- While the Poisson model is a useful approximation, real neurons often deviate from it due to biological constraints:
    - **Refractoriness:** Real neurons have a "refractory period" after firing during which they cannot fire again immediately. This makes very short inter-spike intervals less likely than a Poisson model predicts
    - **Gamma Distribution:** To better account for refractoriness, ISI data is often fitted to a gamma distribution rather than an exponential one:

$$ p[\tau] = \frac{r(r\tau)^k \exp(-r\tau)}{k!} $$
 
- **Variability Sources:** In vitro (in a dish), neurons can fire very regularly, but in vivo (in a living brain), they show the high variability characteristic of Poisson-like processes. This suggests the variability arises from the complex network of synaptic inputs rather than the spike-generation mechanism itself



