import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { Dialog as DialogPrimitive } from 'radix-ui'
import { useEffect, type ReactNode } from 'react'

import { Button } from '@/components/ui/button'

export interface LightboxItem {
  key: string
  src: string
  alt: string
  /** Shared with the thumbnail so the image morphs between the two. */
  layoutId?: string
  title?: ReactNode
  details?: ReactNode
  actions?: ReactNode
}

interface LightboxProps {
  items: LightboxItem[]
  /** Index of the open item, or null when closed. */
  index: number | null
  onIndexChange: (index: number | null) => void
}

/**
 * Full-screen image viewer. The image shares a `layoutId` with its thumbnail,
 * so it grows out of (and shrinks back into) the grid instead of popping in.
 */
export function Lightbox({ items, index, onIndexChange }: LightboxProps) {
  const item = index !== null ? items[index] : undefined
  const open = item !== undefined
  const hasPrev = index !== null && index > 0
  const hasNext = index !== null && index < items.length - 1

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' && hasPrev) onIndexChange(index! - 1)
      if (e.key === 'ArrowRight' && hasNext) onIndexChange(index! + 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, index, hasPrev, hasNext, onIndexChange])

  return (
    <DialogPrimitive.Root open={open} onOpenChange={(o) => !o && onIndexChange(null)}>
      <AnimatePresence>
        {open && (
          <DialogPrimitive.Portal forceMount>
            <DialogPrimitive.Overlay asChild forceMount>
              <motion.div
                className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              />
            </DialogPrimitive.Overlay>
            <DialogPrimitive.Content
              forceMount
              className="fixed inset-0 z-50 flex flex-col outline-none lg:flex-row"
              aria-describedby={undefined}
            >
              <DialogPrimitive.Title className="sr-only">{item.alt}</DialogPrimitive.Title>
              <div className="relative flex min-h-0 flex-1 items-center justify-center p-4 md:p-8">
                <motion.img
                  key={item.key}
                  layoutId={item.layoutId}
                  src={item.src}
                  alt={item.alt}
                  className="max-h-full max-w-full rounded-lg object-contain shadow-2xl"
                  initial={item.layoutId ? false : { opacity: 0, scale: 0.97 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ type: 'spring', stiffness: 260, damping: 30 }}
                />
                {hasPrev && (
                  <Button
                    variant="secondary"
                    size="icon-lg"
                    className="absolute top-1/2 left-3 -translate-y-1/2 rounded-full opacity-80 hover:opacity-100"
                    onClick={() => onIndexChange(index! - 1)}
                    aria-label="Previous image"
                  >
                    <ChevronLeft />
                  </Button>
                )}
                {hasNext && (
                  <Button
                    variant="secondary"
                    size="icon-lg"
                    className="absolute top-1/2 right-3 -translate-y-1/2 rounded-full opacity-80 hover:opacity-100"
                    onClick={() => onIndexChange(index! + 1)}
                    aria-label="Next image"
                  >
                    <ChevronRight />
                  </Button>
                )}
              </div>
              {(item.details || item.actions || item.title) && (
                <motion.aside
                  className="max-h-[40vh] w-full shrink-0 overflow-y-auto border-t bg-background p-5 lg:max-h-none lg:w-96 lg:border-t-0 lg:border-l"
                  initial={{ opacity: 0, x: 24 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 24 }}
                  transition={{ duration: 0.2 }}
                >
                  {item.title && <div className="mb-4 pr-10 font-medium">{item.title}</div>}
                  {item.actions && <div className="mb-5 flex flex-wrap gap-2">{item.actions}</div>}
                  {item.details}
                </motion.aside>
              )}
              <DialogPrimitive.Close asChild>
                <Button
                  variant="secondary"
                  size="icon"
                  className="absolute top-3 right-3 rounded-full"
                  aria-label="Close viewer"
                >
                  <X />
                </Button>
              </DialogPrimitive.Close>
            </DialogPrimitive.Content>
          </DialogPrimitive.Portal>
        )}
      </AnimatePresence>
    </DialogPrimitive.Root>
  )
}

/** Key/value list for image details in the lightbox side panel. */
export function DetailList({ entries }: { entries: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
      {entries
        .filter(([, value]) => value !== undefined && value !== null && value !== '')
        .map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="min-w-0 break-words">{value}</dd>
          </div>
        ))}
    </dl>
  )
}
