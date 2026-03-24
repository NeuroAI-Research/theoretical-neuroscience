# 7 Network Models

## 7.2 Firing-Rate Models

- Total Synaptic Current ($I_s$)
    - The total current delivered to the soma is modeled by convolving the presynaptic spike trains with a **synaptic kernel** $K_s(t)$, which describes the time course of a single synaptic response

- **Synaptic Current from Spikes:** 
    - For a presynaptic neuron $b$ with weight $w_b$ and firing times $t_i$
    - $\rho_b(\tau)$ is the neural response function (a sum of Dirac $\delta$ functions)

$$ w_b \sum_{t_i < t} K_s(t - t_i) = w_b \int_{-\infty}^t d\tau \, K_s(t - \tau) \rho_b(\tau) $$

- **Transition to Firing Rates:** 
    - By replacing $\rho_b(\tau)$ with the continuous firing rate $u_b(\tau)$, the total current from all $N_u$ inputs becomes:

$$ I_s = \sum_{b=1}^{N_u} w_b \int_{-\infty}^t d\tau \, K_s(t - \tau) u_b(\tau) $$

- **Differential Form:** 
    - If $K_s(t)$ is an exponential ($e^{-t/\tau_s}/\tau_s$), $I_s$ can be described by:
    - This equation shows that the current $I_s$ relaxes toward the weighted sum of inputs with a time constant $\tau_s$

$$ \tau_s \frac{dI_s}{dt} = -I_s + \sum w_b u_b = -I_s + \mathbf{w} \cdot \mathbf{u} $$

- Postsynaptic Firing Rate ($v$)
    - The firing rate is determined by an **activation function** $F(I_s)$ 
    - **Threshold Linear Function:** A common choice is the **ReLU**:
    - where $\gamma$ is the firing threshold

$$ F(I_s) = [I_s - \gamma]_+ $$

- **Steady-State Rate:** For constant inputs, the output rate is simply $v_\infty = F(\mathbf{w} \cdot \mathbf{u})$

- Firing-Rate Dynamics
    - There are two main ways to model how the rate $v$ changes over time:

- **Current-Driven Dynamics:** Assumes the firing rate follows the current instantaneously:

$$ \tau_s \frac{dI_s}{dt} = -I_s + \mathbf{w} \cdot \mathbf{u} \quad \text{with} \quad v = F(I_s) $$

- **Rate-Driven Dynamics:** Assumes the firing rate is a low-pass filtered version of the input, which is more common for network analysis:
    - Here, $\tau_r$ is a time constant often related to (but typically smaller than) the membrane time constant.

$$ \tau_r \frac{dv}{dt} = -v + F(\mathbf{w} \cdot \mathbf{u}) $$

- Network Architectures
    - These equations scale to larger populations using matrices:
    - **Recurrent Network:** Including a weight matrix $\mathbf{M}$ for connections between output neurons:

$$ \tau_r \frac{d\mathbf{v}}{dt} = -\mathbf{v} + F( \underbrace{\mathbf{W} \cdot \mathbf{u} }_{\text{Feedforward}} + \underbrace{\mathbf{M} \cdot \mathbf{v} }_{\text{Recurrent}} ) $$

- Recurrence
    - In the text (Biophysical/Rate Models): Recurrence refers to Lateral Connectivity or "Intra-layer" connections. It describes neurons in the same population (e.g., a cortical column) talking to each other. The focus is on how these connections amplify or suppress signals within the same moment in time to reach an equilibrium.
    - In modern ANNs (RNNs/LSTMs): Recurrence refers to Temporal Connectivity. It describes a layer talking to itself in the future. The focus is on Memory—carrying information from time-step $T$ to $T+1$.

- Continuously Labeled Networks
    - In previous equations, $v_i$ referred to the $i$-th neuron. 
    - In a Continuously Labeled Network, we assume neurons are arranged according to a property, such as:
        - Orientation: The angle of a line a neuron "likes" to see ($\theta \in [0, \pi]$).
        - Spatial Location: The $(x, y)$ coordinate on the retina or the body.
    - Instead of a vector of discrete rates, we now have a Function $v(\theta)$
    - In the discrete model, we used a sum $\sum M_{ij} v_j$
    - In the continuous model, this becomes:
    - $v(\theta')$: The activity of a neuron at position $\theta'$
    - $M(\theta, \theta')$: The strength of the connection between a neuron at position $\theta$ and a neuron at position $\theta'$
    - The Integral: This represents the total input to the neuron at $\theta$ from every other possible orientation in the entire population

$$ \tau_r \frac{dv(\theta)}{dt} = -v(\theta) + F\left( \int d\theta' [W(\theta, \theta')u(\theta') + M(\theta, \theta')v(\theta')] \right) $$

- Simulating 100 billion individual neurons is computationally nearly impossible. However, if we treat the cortex as a continuous sheet, we can use Calculus (specifically Integro-Differential Equations) to predict the behavior of the whole system without simulating every cell.
    - We can solve for Stationary States: Where the activity forms a "bump" (representing a memory or a specific perception).
    - We can predict Waves: How an itch or a visual flash "spreads" across the brain's surface.
