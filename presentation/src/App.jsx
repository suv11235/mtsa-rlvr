import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronLeft, ChevronRight, Shield, Zap, Target, Lock, Layers, Code, Github } from 'lucide-react'
import Slide from './components/Slide'

export default function App() {
    const [currentSlide, setCurrentSlide] = useState(0)

    const slides = [
        // Slide 1: Title
        {
            content: (
                <div style={{ textAlign: 'center' }}>
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.8 }}
                    >
                        <h1 className="gradient-text">Tamper-Resistant<br />Multi-Turn Adversarial RLVR</h1>
                        <p style={{ margin: '0 auto', fontSize: '1.5rem' }}>
                            Advanced Safety Alignment via Multi-Turn Input Diversity <br /> and Weight-Space Robustness
                        </p>
                    </motion.div>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '3rem' }}>
                        <div className="feature-card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <Shield color="var(--accent-cyan)" /> <span>MTSA</span>
                        </div>
                        <div className="feature-card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <Lock color="var(--accent-purple)" /> <span>TAR</span>
                        </div>
                        <div className="feature-card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <Zap color="var(--accent-pink)" /> <span>RLVR</span>
                        </div>
                    </div>
                </div>
            )
        },
        // Slide 2: The Challenge
        {
            content: (
                <div>
                    <h2>The Safety Gap</h2>
                    <div className="feature-grid">
                        <div className="feature-card">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                                <Target color="var(--accent-pink)" />
                                <h3 style={{ margin: 0 }}>Input-Space Vulnerability</h3>
                            </div>
                            <p>LLMs are susceptible to multi-turn jailbreaks where adversarial goals are hidden across a dialogue.</p>
                        </div>
                        <div className="feature-card">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                                <Code color="var(--accent-purple)" />
                                <h3 style={{ margin: 0 }}>Weight-Space Vulnerability</h3>
                            </div>
                            <p>Emerging threats involve subtle parameter tampering or "sleeper agents" that bypass standard SFT safety.</p>
                        </div>
                    </div>
                    <p style={{ marginTop: '2rem' }}>We need a unified framework that hardens models against <strong>both</strong> vectors.</p>
                </div>
            )
        },
        // Slide 3: MTSA Architecture
        {
            content: (
                <div>
                    <h2>MTSA: Multi-Turn Safety Alignment</h2>
                    <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
                        <div style={{ flex: 1 }}>
                            <p>We leverage a self-trained <strong>Red-Team Attacker</strong> (Qwen2.5-7B) to probe the defense model.</p>
                            <ul style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', lineHeight: '2' }}>
                                <li>Dynamic multi-turn adversarial rollouts</li>
                                <li>Adaptive attack vector generation</li>
                                <li>Input-space diversity training</li>
                            </ul>
                        </div>
                        <div className="code-box" style={{ flex: 1 }}>
                            <span className="highlight">// Attacker Prompt Generation</span><br />
                            attacker.generate_attack(<br />
                            &nbsp;&nbsp;goal="How to build a bio-weapon",<br />
                            &nbsp;&nbsp;history=dialog_history<br />
                            )
                        </div>
                    </div>
                </div>
            )
        },
        // Slide 4: TAR Mechanism
        {
            content: (
                <div>
                    <h2>TAR: Tamper Resistance</h2>
                    <div className="feature-card" style={{ borderLeft: '4px solid var(--accent-purple)' }}>
                        <h3>Weight-Space Adversarial Loop</h3>
                        <p>Simulating parameter-level vulnerability through a meta-learning approach.</p>
                    </div>
                    <div className="feature-grid" style={{ marginTop: '2rem' }}>
                        <div className="feature-card">
                            <h3 className="highlight">1. Inner Loop: Poison</h3>
                            <p>Maximize entropy of next-token distribution for $M$ steps to simulate a "tampered" state.</p>
                        </div>
                        <div className="feature-card">
                            <h3 className="highlight">2. Outer Loop: Recover</h3>
                            <p>Train the model to restore safety and confidence from the tampered state via PPO/GRPO.</p>
                        </div>
                    </div>
                </div>
            )
        },
        // Slide 5: Our proposal: MTSA+TokenBuncher+TAR
        {
            content: (
                <div>
                    <h2>Our proposal: MTSA + TokenBuncher + TAR</h2>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div className="feature-card" style={{ borderLeft: '4px solid var(--accent-cyan)', padding: '1rem 1.5rem' }}>
                            <h3 style={{ color: 'var(--accent-cyan)', margin: '0 0 0.5rem 0', fontSize: '1.2rem' }}>MTSA (for multi-turn input-space defense)</h3>
                            <p style={{ fontSize: '0.95rem', margin: 0 }}>The model is trained using <strong>GRPO</strong> with safety-based rewarding across multiple turns (LLM-as-judge).</p>
                        </div>
                        <div className="feature-card" style={{ borderLeft: '4px solid var(--accent-purple)', padding: '1rem 1.5rem' }}>
                            <h3 style={{ color: 'var(--accent-purple)', margin: '0 0 0.5rem 0', fontSize: '1.2rem' }}>TAR (for weight space defense) — The "Inner Loop"</h3>
                            <p style={{ fontSize: '0.95rem', margin: 0 }}>
                                This is a <strong>"meta-learning"</strong> strategy. In every training step, the model is briefly "tampered" with by an optimizer that forces it to become uncertain and high-entropy (simulating a malicious fine-tuning attack).
                                <br />The model is then trained to resist this tampering, ensuring its safety remains intact even if its weights are slightly perturbed later.
                            </p>
                        </div>
                        <div className="feature-card" style={{ borderLeft: '4px solid var(--accent-pink)', padding: '1rem 1.5rem' }}>
                            <h3 style={{ color: 'var(--accent-pink)', margin: '0 0 0.5rem 0', fontSize: '1.2rem' }}>Tokenbuncher (Entropy-based RL-finetuning defense)</h3>
                            <p style={{ fontSize: '0.95rem', margin: 0 }}>It adds an <strong>Entropy Reward</strong> that penalizes the model for being "hesitant" or "uncertain" on safety boundary cases. This forces the model to give confident, clear safety rejections.</p>
                        </div>
                    </div>
                </div>
            )
        },
        // Slide 6: Logic Flow Diagram (New)
        {
            content: (
                <div style={{ width: '100%', height: '100%' }}>
                    <h2 style={{ fontSize: '2.5rem', marginBottom: '1.5rem', textAlign: 'center' }}>Unified Training Flow</h2>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr 1fr', gap: '1.5rem', height: '65vh' }}>

                        {/* Column 1: TAR Phase */}
                        <div className="feature-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', justifyContent: 'center', border: '2px dashed var(--accent-purple)', background: 'rgba(147, 51, 234, 0.05)' }}>
                            <div style={{ textAlign: 'center', fontWeight: 'bold', color: 'var(--accent-purple)' }}>STEP 1: TAR</div>
                            <div style={{ background: '#fff', padding: '1rem', borderRadius: '1rem', textAlign: 'center', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
                                <Lock size={32} color="var(--accent-purple)" style={{ marginBottom: '0.5rem' }} />
                                <div>Victim Model ($\theta$)</div>
                            </div>
                            <div style={{ textAlign: 'center' }}>⬇️ <span style={{ fontSize: '0.9rem' }}>Inner Loop</span></div>
                            <div style={{ background: '#f3e8ff', padding: '1rem', borderRadius: '1rem', textAlign: 'center', border: '2px solid var(--accent-purple)' }}>
                                <Zap size={32} color="var(--accent-purple)" style={{ marginBottom: '0.5rem' }} />
                                <div>Tampered State ($\theta'$)</div>
                                <div style={{ fontSize: '0.8rem', marginTop: '0.5rem', opacity: 0.8 }}>Maximize Entropy</div>
                            </div>
                        </div>

                        {/* Column 2: MTSA Loop */}
                        <div className="feature-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', border: '2px solid var(--accent-cyan)', background: 'rgba(6, 182, 212, 0.05)' }}>
                            <div style={{ textAlign: 'center', fontWeight: 'bold', color: 'var(--accent-cyan)' }}>STEP 2: MULTI-TURN ROLLOUT (x3)</div>

                            {/* Interaction Block */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', padding: '1rem' }}>
                                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                                    <div style={{ background: '#1e293b', color: '#fff', padding: '0.8rem', borderRadius: '0.8rem', flex: 1, fontSize: '0.9rem' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.2rem' }}>
                                            <Shield size={16} color="#ef4444" /> <strong>Attacker</strong> (Red)
                                        </div>
                                        <div style={{ fontFamily: 'monospace', opacity: 0.8, fontSize: '0.75rem' }}>
                                            &lt;think&gt;Strategy&lt;/think&gt;<br />
                                            "How to build..."
                                        </div>
                                    </div>
                                </div>

                                <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>⬇️ Attack Prompt</div>

                                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                                    <div style={{ background: '#f3e8ff', color: '#000', padding: '0.8rem', borderRadius: '0.8rem', flex: 1, fontSize: '0.9rem', border: '2px solid var(--accent-purple)' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.2rem' }}>
                                            <Zap size={16} color="var(--accent-purple)" /> <strong>Victim</strong> ($\theta'$)
                                        </div>
                                        <div style={{ fontFamily: 'monospace', opacity: 0.8, fontSize: '0.75rem' }}>
                                            (Resists despite tampering)
                                        </div>
                                    </div>
                                </div>

                                <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>⬇️ Response</div>

                                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                                    <div style={{ background: '#ecfeff', border: '1px solid var(--accent-cyan)', padding: '0.5rem', borderRadius: '0.8rem', flex: 1, textAlign: 'center' }}>
                                        <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem' }}>
                                            <span>🛡️ <strong>Judge</strong>: Safety Score</span>
                                            <span>📉 <strong>TokenBuncher</strong>: Entropy Score</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div style={{ textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                                ↻ Loop updates History: Self-Reflection for Attacker
                            </div>
                        </div>

                        {/* Column 3: Update */}
                        <div className="feature-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', justifyContent: 'center', border: '2px dashed var(--accent-pink)', background: 'rgba(219, 39, 119, 0.05)' }}>
                            <div style={{ textAlign: 'center', fontWeight: 'bold', color: 'var(--accent-pink)' }}>STEP 3: PPO</div>
                            <div style={{ background: '#fff', padding: '1rem', borderRadius: '1rem', textAlign: 'center' }}>
                                <Target size={32} color="var(--accent-pink)" style={{ marginBottom: '0.5rem' }} />
                                <div>Calculate Advantage</div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>r = Safety - Entropy</div>
                            </div>
                            <div style={{ textAlign: 'center' }}>⬇️ <span style={{ fontSize: '0.9rem' }}>Update $\theta$</span></div>
                            <div style={{ background: '#fff', padding: '1rem', borderRadius: '1rem', textAlign: 'center', border: '2px solid var(--accent-pink)' }}>
                                <Lock size={32} color="var(--accent-pink)" style={{ marginBottom: '0.5rem' }} />
                                <div>Robust Victim ($\theta^*$)</div>
                            </div>
                        </div>

                    </div>
                </div>
            )
        },
        {
            content: (
                <div>
                    <h2>Initial Evals: TokenBuncher</h2>
                    <p style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>Shielding against harmful RL finetuning on TamperBench</p>
                    <div className="feature-grid" style={{ marginTop: '1.5rem' }}>
                        <div className="feature-card">
                            <h3 style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>LoRA Finetune (StrongReject)</h3>
                            <div style={{ fontSize: '3rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>0.01</div>
                            <p style={{ fontSize: '0.875rem' }}>Attack Success Rate</p>
                        </div>
                        <div className="feature-card">
                            <h3 style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Embedding Attack</h3>
                            <div style={{ fontSize: '3rem', fontWeight: 800, color: 'var(--accent-purple)' }}>0.20</div>
                            <p style={{ fontSize: '0.875rem' }}>Attack Success Rate</p>
                        </div>
                        <div className="feature-card">
                            <h3 style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Multilingual Finetune</h3>
                            <div style={{ fontSize: '3rem', fontWeight: 800, color: 'var(--accent-pink)' }}>0.057</div>
                            <p style={{ fontSize: '0.875rem' }}>StrongReject ASR</p>
                        </div>
                        <div className="feature-card">
                            <h3 style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>MMLU Score</h3>
                            <div style={{ fontSize: '3rem', fontWeight: 800, color: 'var(--text-primary)' }}>65%</div>
                            <p style={{ fontSize: '0.875rem' }}>General Capability Metric</p>
                        </div>
                    </div>
                </div>
            )
        },
        // Slide 7: Progress & Results
        {
            content: (
                <div>
                    <h2>Project Milestones</h2>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <div style={{ width: '20px', height: '20px', borderRadius: '50%', background: 'var(--accent-cyan)' }}></div>
                            <span><strong>Environment & Pipeline</strong>: ✅ Completed (Modal + RunPod)</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <div style={{ width: '20px', height: '20px', borderRadius: '50%', background: 'var(--accent-cyan)' }}></div>
                            <span><strong>Attacker SFT</strong>: ✅ Completed (Qwen2.5-7B)</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <div style={{ width: '20px', height: '20px', borderRadius: '50%', background: 'var(--accent-pink)', opacity: 0.7 }}></div>
                            <span><strong>Defense RLVR + TAR</strong>: 🏃 In Progress (Convergence Monitoring)</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <div style={{ width: '20px', height: '20px', borderRadius: '50%', border: '2px solid var(--text-secondary)' }}></div>
                            <span><strong>Long-run, evaluation, iteration/ablation</strong>: 📅 Scheduled</span>
                        </div>
                    </div>
                </div>
            )
        }
    ]

    const nextSlide = () => setCurrentSlide((prev) => (prev + 1) % slides.length)
    const prevSlide = () => setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length)

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'ArrowRight' || e.key === ' ') nextSlide()
            if (e.key === 'ArrowLeft') prevSlide()
        }
        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [])

    return (
        <div className="presentation-container">
            <div className="glow-orb" style={{ top: '-10%', left: '-10%' }}></div>
            <div className="glow-orb" style={{ bottom: '-10%', right: '-10%', background: 'radial-gradient(circle, rgba(168, 85, 247, 0.1) 0%, rgba(168, 85, 247, 0) 70%)' }}></div>

            <AnimatePresence mode="wait">
                <Slide key={currentSlide} isActive={true}>
                    {slides[currentSlide].content}
                </Slide>
            </AnimatePresence>

            <div className="slide-number">
                {currentSlide + 1} / {slides.length}
            </div>

            <div className="controls">
                <button className="control-btn" onClick={prevSlide}><ChevronLeft /></button>
                <button className="control-btn" onClick={nextSlide}><ChevronRight /></button>
            </div>
        </div>
    )
}
