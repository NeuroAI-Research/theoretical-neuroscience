# 9 Classical Conditioning and Reinforcement Learning

## 9.2 Classical Conditioning

- The Biology: From Pavlov to Dopamine
    - Classical conditioning is the process by which animals learn to associate stimuli with rewards or punishments
    - The Behavioral Foundation
        - **Unconditioned Stimulus (US):** A stimulus that naturally triggers a response, such as food
        - **Unconditioned Response (UR):** The natural reaction to a US, such as salivating when receiving food
        - **Conditioned Stimulus (CS):** A neutral cue (like a bell) that, through learning, begins to elicit a response
        - **Conditioned Response (CR):** The learned response to the CS, marking the animal's expectation of a reward

- The Neural Mechanism: Dopamine in the VTA
    - Computational models suggest that the **midbrain dopamine system**, specifically the **Ventral Tegmental Area (VTA)**, encodes the **prediction error ($\delta$)**
    - **Early Training:** Dopaminergic neurons fire vigorously upon receiving an unexpected reward
    - **Late Training:** After learning, neurons fire at the time of the **stimulus** (the predictor) but stay at baseline when the reward actually arrives, as it is no longer a "surprise"
    - **Omission:** If an expected reward is denied, dopamine activity is inhibited below its basal firing rate at the exact time the reward was predicted

- A. The Rescorla-Wagner Rule (The Delta Rule)
    - This model provides a concise account of how animals learn a linear prediction of reward based on the presence of stimuli
    - $v$: The expected reward
    - $u$: A binary stimulus variable (1 if present, 0 if absent)
    - $w$: The strength of the association (weight)
    - $\delta$: The prediction error
    - $\epsilon$: The learning rate

$$ v = w \cdot u \\[5pt]
\delta = r - v \\[5pt]
w \rightarrow w + \epsilon \delta u $$ 

- B. Temporal Difference (TD) Learning
    - TD learning **extends** the Rescorla-Wagner rule to account for the **timing** of rewards within a trial
    - The model treats $v(t)$ as a prediction of the **total future reward** expected from time $t$ until the end of the trial ($T$)

    $$ v(t) \approx \langle \sum_{\tau=0}^{T-t} r(t+\tau) \rangle $$

    - $v(t)$ is calculated using a discrete time version of a linear filter
    
    $$ v(t) = \sum_{\tau=0} w(\tau) u(t-\tau) $$

    - Because future rewards are unknown, the model uses the next step's prediction ($v(t+1)$) to calculate the error
    
    $$ \delta(t) = r(t) + v(t+1) - v(t) \\[5pt]
    w(\tau) \rightarrow w(\tau) + \epsilon \delta(t) u(t-\tau)$$

- Explaining Behavioral Phenomena

| Paradigm | Behavioral Result | Mathematical Explanation |
| :--- | :--- | :--- |
| **Blocking** | Pre-training stimulus $s_1$ prevents learning stimulus $s_2$ | Since $w_1 = r$, the prediction $v$ is already correct; thus $\delta = 0$ and $w_2$ never increases |
| **Extinction** | A learned response decays when the reward is removed | If $r=0$, then $\delta$ is negative, causing the weight $w$ to exponentially decay toward 0 |
| **Overshadowing** | One stimulus is learned more strongly than another when paired | This is modeled by giving the two stimuli different learning rates ($\epsilon$) |
| **Secondary Conditioning** | $s_2$ predicts $s_1$, which predicts reward. $s_2$ now evokes reward expectation | **TD Learning** explains this: the positive $\delta$ spike at the time of $s_1$ acts as a reward to drive the learning of $s_2$ |
