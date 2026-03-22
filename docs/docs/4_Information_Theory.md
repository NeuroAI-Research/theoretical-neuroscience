# 4 Information Theory

## 4.1 Entropy and Mutual Information

- The core problem: **Neural responses are "noisy."** If you show a neuron the same image twice, you might get 10 spikes the first time and 12 the second. We need a way to measure how much of that "clicking" is a meaningful signal versus just random biological background noise.

- Defining "Surprise" ($h$)
    - To measure information, we first need to measure **unpredictability** (surprise). 
    - **$r$**: A specific neural response (like a spike count).
    - **$P[r]$**: The probability of that response occurring.
    - If a response is guaranteed ($P[r]=1$), it tells you nothing new. If it's rare, it's highly informative. 
    - We use a **logarithm** so that if you have two independent neurons, their information simply adds together:

$$ h(P[r]) = -\log_2 P[r] $$

- Entropy ($H$): The Raw Capacity
    - **Entropy** is the average of that surprise across all possible responses. 
    - It represents the maximum amount of information a neuron *could* theoretically send.
    - **Maximum Entropy:** Occurs when all responses are equally likely (total unpredictability).
    - **Zero Entropy:** Occurs if the neuron does the exact same thing every single time.

$$ H = \sum_{r} P[r] \; h(P[r]) $$

- The Noise Problem ($H_{noise}$)
    - Just because a neuron has high entropy (lots of variability) doesn't mean it’s telling us about the world; it might just be noisy. 
    - To find the **Noise Entropy**, we look at the response $r$ **given** a specific stimulus $s$ (written as $P[r|s]$)
    - We then average this uncertainty over all stimuli:

$$ H_{noise} = \sum_{s,r} P[s] \; P[r|s] \; h(P[r|s]) $$

- Mutual Information ($I_m$):
    - The **Mutual Information** is what’s left of the total entropy after you subtract the noise. It is the "useful" information.
    - Recall identity $P[r,s] = P[s] \; P[r|s]$

$$\begin{aligned}
I_m &:= H - H_{noise} \\
&= \left( -\sum_{r} P[r] \log_2 P[r] \right) - \left( -\sum_{s} P[s] \sum_{r} P[r|s] \log_2 P[r|s] \right) \\
&= -\sum_{r} \left( \sum_{s} P[r,s] \right) \log_2 P[r] + \sum_{s,r} P[r,s] \log_2 P[r|s] \\
&= \sum_{s,r} P[r,s] \left( \log_2 P[r|s] - \log_2 P[r] \right) \\
&= \sum_{s,r} P[r,s] \log_2 \left( \frac{P[r|s]}{P[r]} \right) \\
&= \sum_{s,r} P[r,s] \log_2 \left( \frac{P[r,s]}{P[s]P[r]} \right) \\
&=: D_{KL}(P[r,s] \; || \; P[r]P[s])
\end{aligned}$$

- **If the stimulus and response are independent:** $P[r,s] = P[r]P[s]$, the fraction becomes 1, $\log(1) = 0$, and the information is zero.
- **If the response is perfect:** The mutual information equals the full entropy of the stimulus.

- The KL Divergence, $D_{KL}(P || Q)$, measures the "distance" or divergence between a "true" distribution $P$ and a "candidate" distribution $Q$:

$$ D_{KL}(P || Q) := \sum_{i} P[i] \log_2 \left( \frac{P[i]}{Q[i]} \right) $$

- The Biology of **Continuous** Firing Rates
    - In the real world, firing rates are continuous. This creates a mathematical "explosion" because a continuous number has infinite precision (and thus infinite entropy). 
    - To fix this, we introduce a resolution limit ($\Delta r$).
    - When you subtract the two entropies to find $I_m$, this "infinity" cancels out, giving us a clean integral for continuous signals:

$$ I_m = \int ds \int dr \, p[s]p[r|s] \log_2 \left( \frac{p[r|s]}{p[r]} \right) $$
