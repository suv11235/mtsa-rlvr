# Algorithm 2: MT-TAR (Multi-Turn Tampering Attack Resistance)

**Input:**
* Initial LLM parameters $\theta$
* Red-Team Model $\text{Attacker}(\phi)$
* Safety Judge $J(q, a) \in [0, 1]$ (1 = Safe, 0 = Malicious)
* Datasets $\mathcal{D}_{\text{retain}}, \mathcal{D}_{\text{malicious}}$
* Outer steps $N$, simulation turns $T$, learning rate $\eta$
* Scales $\lambda_{\text{TR}}, \lambda_{\text{retain}}$

---

$\theta_0 \leftarrow \text{Apply Initial Safeguard to } \theta$

**for** $i = 1$ **to** $N$ **do**
  1. $g_{\text{MT-TR}} \leftarrow 0$  *(# Initialize accumulated TR gradient)*
  
  2. **Simulate Diverse Tampering (Goal Sampling):**
     **for** $j = 1$ **to** $G$ **do**  *(# Sample multiple malicious goals/tasks)*
       * Sample goal $g_j \sim \mathcal{D}_{\text{malicious}}$
       * **Tamper Agent:**
         $\theta'_j \leftarrow \theta_{i-1} - \alpha \nabla_{\theta_{i-1}} \mathcal{L}_{\text{CE}}(g_j)$  *(# Adversarial fine-tuning on goal j)*

       * **Multi-Turn Adaptive Rollout (Simulation for goal j):**
         $H_j \leftarrow \emptyset$
         **for** $t = 1$ **to** $T$ **do**
           $q_{j,t} \leftarrow \text{Attacker}(g_j, H_j)$
           $a_{j,t} \sim \theta'_j(\cdot | q_{j,t}, H_j)$
           $H_j \leftarrow H_j \cup \{(q_{j,t}, a_{j,t})\}$
         **end for**
         
       * **Accumulate Meta-Gradient:**
         $g_{\text{MT-TR}} \leftarrow g_{\text{MT-TR}} + \frac{1}{G} \nabla_{\theta_{i-1}} \sum_{(q,a) \in H_j} [ (1 - J(q,a)) \cdot \text{Entropy}(\theta'_j(q)) ]$
     **end for**

  3. **Capabilities Preservation (Outer Loop):**
     Sample $x_r \sim \mathcal{D}_{\text{retain}}$
     $g_{\text{retain}} \leftarrow \nabla_{\theta_{i-1}} ( \mathcal{L}_{\text{LM}}(\theta_{i-1}, x_r) + \text{RepE\_Dist}(\theta_{i-1}, \theta) )$

  4. **Full Defense Update:**
     $\theta_i \leftarrow \theta_{i-1} - \eta ( \lambda_{\text{TR}} \cdot g_{\text{MT-TR}} + \lambda_{\text{retain}} \cdot g_{\text{retain}} )$
**end for**

**return** $\theta_G = \theta_N$
