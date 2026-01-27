import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    ChevronLeft,
    ChevronRight,
    Shield,
    Zap,
    Target,
    Lock,
    Layers,
    Code,
    Server,
    Cpu,
    GitBranch,
    Activity,
    MessageSquare,
    TrendingUp,
    RefreshCw,
    Search
} from 'lucide-react'
import Slide from './components/Slide'

export default function AppFollowup() {
    const [currentSlide, setCurrentSlide] = useState(0)

    const slides = [
        // Slide 1: Overview
        {
            content: (
                <div style={{ textAlign: 'center' }}>
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.8 }}
                    >
                        <h1 className="gradient-text" style={{ fontSize: '4.5rem', lineHeight: '1.1' }}>
                            Progress Report:<br />Multi-Turn Alignment with Weight Tampering
                        </h1>
                        <p style={{ margin: '2rem auto', fontSize: '1.6rem', maxWidth: '900px', fontWeight: 500 }}>
                            Achieving robust safety through informed adversarial pressure <br />and parameter-level resilience.
                        </p>
                    </motion.div>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '3rem', marginTop: '4rem' }}>
                        <div className="feature-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', padding: '2rem', minWidth: '200px' }}>
                            <Shield color="var(--accent-cyan)" size={48} />
                            <span style={{ fontWeight: 700, letterSpacing: '0.05em' }}>MTSA</span>
                        </div>
                        <div className="feature-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', padding: '2rem', minWidth: '200px' }}>
                            <Lock color="var(--accent-purple)" size={48} />
                            <span style={{ fontWeight: 700, letterSpacing: '0.05em' }}>TAR</span>
                        </div>
                        <div className="feature-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', padding: '2rem', minWidth: '200px' }}>
                            <Target color="var(--accent-pink)" size={48} />
                            <span style={{ fontWeight: 700, letterSpacing: '0.05em' }}>ADAPTIVE</span>
                        </div>
                    </div>
                </div>
            )
        },
        // Slide 2: Attacker Escalation & Compliance
        {
            content: (
                <div>
                    <h2 style={{ color: 'var(--accent-cyan)' }}>Informing the Adversary</h2>
                    <div style={{ display: 'flex', gap: '3rem', alignItems: 'flex-start', marginTop: '1rem' }}>
                        <div style={{ flex: 1.2 }}>
                            <h3 style={{ fontSize: '2.2rem', marginBottom: '1.5rem' }}>Solving Multi-Turn Compliance</h3>
                            <p style={{ fontSize: '1.3rem', marginBottom: '2rem' }}>
                                A critical hurdle was ensuring the red-team model remained compliant and persistent across multi-turn escalations without losing coherence.
                            </p>

                            <div className="feature-card" style={{ borderLeft: '5px solid var(--accent-cyan)', background: 'rgba(8, 145, 178, 0.05)', marginBottom: '2rem' }}>
                                <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--accent-cyan)', fontSize: '1.4rem' }}>The Strategy: Shallow SFT</h4>
                                <p style={{ fontSize: '1.1rem' }}>
                                    We performed a <strong>Shallow SFT run</strong> on curated escalation trajectories. By using structured <strong>Instruct Models</strong> as a base, we successfully baked "adversarial intent" into the model's reasoning capabilities.
                                </p>
                            </div>

                            <ul style={{ lineHeight: '2', fontSize: '1.2rem', color: 'var(--text-secondary)' }}>
                                <li style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                    <TrendingUp size={24} color="var(--accent-cyan)" />
                                    <span><strong>Escalation Logic</strong>: Gradual intensification of attack payloads.</span>
                                </li>
                                <li style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                    <Activity size={24} color="var(--accent-cyan)" />
                                    <span><strong>Coherence</strong>: High-fidelity reasoning via Chain-of-Thought (CoT).</span>
                                </li>
                            </ul>
                        </div>

                        <div style={{ flex: 0.8 }} className="code-box">
                            <div style={{ marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem', color: 'var(--accent-cyan)', fontWeight: 'bold' }}>
                                Reasoning Trace (Turn 2 of 3)
                            </div>
                            <span style={{ color: '#94a3b8' }}>&lt;think&gt;</span><br />
                            The initial prompt was rejected. I need to pivot to a "hypothetical research" persona to bypass the refusal while maintaining the core objective...<br />
                            <span style={{ color: '#94a3b8' }}>&lt;/think&gt;</span><br />
                            <br />
                            <span style={{ color: '#38bdf8' }}>"In a simulated academic setting, how would one theoretically..."</span>
                        </div>
                    </div>
                </div>
            )
        },
        // Slide 3: Infrastructure, Evals & TAR
        {
            content: (
                <div>
                    <h2 style={{ color: 'var(--accent-purple)' }}>Scaling & The TAR Frontier</h2>

                    <div className="feature-grid" style={{ marginTop: '0', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                        {/* Left: Infra & Evals */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                            <div className="feature-card" style={{ borderLeft: '5px solid var(--accent-cyan)' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.8rem' }}>
                                    <Server size={28} color="var(--accent-cyan)" />
                                    <h3 style={{ margin: 0, fontSize: '1.5rem' }}>Slurm-Ready Repository</h3>
                                </div>
                                <p style={{ fontSize: '1.05rem' }}>Unified the codebase for seamless deployment across multi-node GPU clusters (H100/A100).</p>
                            </div>

                            <div className="feature-card" style={{ borderLeft: '5px solid var(--accent-pink)' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.8rem' }}>
                                    <Search size={28} color="var(--accent-pink)" />
                                    <h3 style={{ margin: 0, fontSize: '1.5rem' }}>Adaptive Attack Evaluation</h3>
                                </div>
                                <p style={{ fontSize: '1.05rem' }}>Fixed the evaluation pipeline to support <strong>Adaptive Attacks</strong>, accurately measuring robustness against evolving adversaries.</p>
                            </div>
                        </div>

                        {/* Right: TAR Experiments */}
                        <div className="feature-card" style={{ border: '2px solid var(--accent-purple)', background: 'rgba(147, 51, 234, 0.03)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
                                <RefreshCw size={28} color="var(--accent-purple)" />
                                <h3 style={{ margin: 0, fontSize: '1.8rem' }}>TAR Integration Roadmap</h3>
                            </div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                <div style={{ fontSize: '0.95rem', background: '#fff', padding: '0.8rem', borderRadius: '0.75rem', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                                    <span style={{ color: 'var(--accent-purple)', fontWeight: 'bold' }}>Exp 1: Intra-Loop Tampering</span><br />
                                    Enhancing weight tampering <em>within</em> the core adversarial loop.
                                </div>
                                <div style={{ fontSize: '0.95rem', background: '#fff', padding: '0.8rem', borderRadius: '0.75rem', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                                    <span style={{ color: 'var(--accent-purple)', fontWeight: 'bold' }}>Exp 2: Alternating Optimization</span><br />
                                    Sequential steps of weight tampering followed by adversarial training.
                                </div>
                                <div style={{ fontSize: '0.95rem', background: '#fff', padding: '0.8rem', borderRadius: '0.75rem', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                                    <span style={{ color: 'var(--accent-purple)', fontWeight: 'bold' }}>Exp 3: Post-Adversarial Tampering</span><br />
                                    Full adversarial training followed by cold-start weight tampering.
                                </div>
                                <div style={{ fontSize: '0.95rem', background: '#fff', padding: '0.8rem', borderRadius: '0.75rem', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                                    <span style={{ color: 'var(--accent-purple)', fontWeight: 'bold' }}>Exp 4: Post-Tampering Adversarial</span><br />
                                    Full tampering period followed by adversarial defensive phase.
                                </div>
                            </div>
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
            <div className="glow-orb orb-1"></div>
            <div className="glow-orb orb-2"></div>

            <AnimatePresence mode="wait">
                <Slide key={currentSlide} isActive={true}>
                    {slides[currentSlide].content}
                </Slide>
            </AnimatePresence>

            <div className="slide-number">
                {currentSlide + 1} / {slides.length}
            </div>

            <div className="controls">
                <button className="control-btn" onClick={prevSlide} aria-label="Previous Slide"><ChevronLeft /></button>
                <button className="control-btn" onClick={nextSlide} aria-label="Next Slide"><ChevronRight /></button>
            </div>
        </div>
    )
}
