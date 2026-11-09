import { motion, type HTMLMotionProps } from 'motion/react'
import { useState } from 'react'

/** An image that fades in from blurred to sharp once it has loaded. */
export function RevealImage({ onLoad, ...props }: HTMLMotionProps<'img'>) {
  const [loaded, setLoaded] = useState(false)
  return (
    <motion.img
      decoding="async"
      initial={false}
      animate={
        loaded ? { opacity: 1, filter: 'blur(0px)', scale: 1 } : { opacity: 0, filter: 'blur(16px)', scale: 1.02 }
      }
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      onLoad={(e) => {
        setLoaded(true)
        onLoad?.(e)
      }}
      {...props}
    />
  )
}
