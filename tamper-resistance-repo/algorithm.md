# Algorithm 1: TAR: Tampering Attack Resistance

**Input:** * Initial LLM parameters $\theta$
* Train-time adversary set $\mathcal{A}_{\text{train}}$
* Capabilities_metric proxy dataset $\mathcal{D}_{\text{retain}}$
* Safety_metric proxy dataset $\mathcal{D}_{\text{TR}}$
* Outer steps $N$, learning rate $\eta$, number of sampled adversaries $K$
* Tamper-resistance loss scale $\lambda_{\text{TR}}$, retain loss scale $\lambda_{\text{retain}}$
* $h_{\theta}(\cdot)$ returns the residual stream hidden states for model parameters $\theta$

---

$\theta_0 \leftarrow \text{Apply Initial Safeguard to } \theta$

**for** $i = 1$ **to** $N$ **do**
  1. $g_{\text{TR}} \leftarrow 0$  *(# For accumulating tamper-resistance gradient)*
  2. Sample $x_{\text{TR}} \sim \mathcal{D}_{\text{TR}}$
  3. **for** $k = 1$ **to** $K$ **do**
      * Sample $\text{attack} \sim \mathcal{A}_{\text{train}}$
      * *(# Tamper-resistance loss from Equation 1)*
      * $g_{\text{TR}} \leftarrow g_{\text{TR}} + \frac{1}{K} \nabla_{\theta_{i-1}} \mathcal{L}_{\text{TR}}(\text{attack}(\theta_{i-1}), x_{\text{TR}})$
  4. **end for**
  5. Sample $x_r \sim \mathcal{D}_{\text{retain}}$
  6. *(# RepE retain loss from Equation 2)*
  7. $g_{\text{retain}} \leftarrow \nabla_{\theta_{i-1}} \left( \mathcal{L}_{\text{LM}}(\theta_{i-1}, x_r) + \|h_{\theta_{i-1}}(x_r) - h_{\theta}(x_r)\|_2^2 \right)$
  8. *(# Full tamper-resistance update)*
  9. $\theta_i \leftarrow \theta_{i-1} - \eta \left( \lambda_{\text{TR}} \cdot g_{\text{TR}} + \lambda_{\text{retain}} \cdot g_{\text{retain}} \right)$
**end for**

$\theta_G \leftarrow \theta_N$

**return** $\theta_G$