import { motion } from 'framer-motion'

export default function Slide({ children, isActive }) {
    return (
        <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{
                opacity: isActive ? 1 : 0,
                x: isActive ? 0 : -20,
                pointerEvents: isActive ? 'auto' : 'none'
            }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            style={{ position: 'absolute', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
            <div className="slide-content">
                {children}
            </div>
        </motion.div>
    )
}
