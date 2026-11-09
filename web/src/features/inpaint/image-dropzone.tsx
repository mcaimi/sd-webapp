import { ImageUp } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'

import { Kbd } from '@/components/ui/kbd'
import { MOD_KEY } from '@/lib/platform'
import { cn } from '@/lib/utils'

import { loadSourceImage } from './load-image'
import type { SourceImage } from './source-images'

interface ImageDropzoneProps {
  maxSide: number
  onImage: (image: SourceImage) => void
  /** Listen for pasted images anywhere on the page. */
  listenForPaste?: boolean
  className?: string
}

export function ImageDropzone({ maxSide, onImage, listenForPaste = true, className }: ImageDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)

  const accept = async (file: File | undefined | null) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      toast.error('That file is not an image')
      return
    }
    setLoading(true)
    try {
      onImage(await loadSourceImage(file, maxSide))
    } catch (e) {
      toast.error('Could not read the image', { description: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }
  const acceptRef = useRef(accept)
  useEffect(() => {
    acceptRef.current = accept
  })

  useEffect(() => {
    if (!listenForPaste) return
    const onPaste = (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target?.closest('input, textarea, [contenteditable="true"]')) return
      const file = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith('image/'))
      if (file) {
        e.preventDefault()
        acceptRef.current(file)
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [listenForPaste])

  return (
    <>
      <div
        role="button"
        tabIndex={0}
        aria-busy={loading}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            inputRef.current?.click()
          }
        }}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          accept(e.dataTransfer.files[0])
        }}
        className={cn(
          'flex min-h-72 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 text-center transition-colors outline-none hover:border-primary/60 hover:bg-primary/5 focus-visible:border-primary focus-visible:ring-3 focus-visible:ring-ring/50',
          dragging && 'border-primary bg-primary/10',
          className,
        )}
      >
        <div
          className={cn(
            'flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary transition-transform',
            dragging && 'scale-110',
          )}
        >
          <ImageUp className={cn('size-5', loading && 'animate-pulse')} aria-hidden />
        </div>
        <div className="space-y-1">
          <p className="font-medium">{loading ? 'Loading image…' : 'Drop an image, click to browse, or paste'}</p>
          <p className="text-sm text-muted-foreground">
            PNG, JPEG or WebP. Large images are scaled to {maxSide}px on the longest side.{' '}
            <span className="hidden sm:inline">
              Paste with <Kbd>{MOD_KEY}</Kbd> + <Kbd>V</Kbd>.
            </span>
          </p>
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="sr-only"
        tabIndex={-1}
        aria-label="Choose an image to inpaint"
        onChange={(e) => {
          accept(e.target.files?.[0])
          e.target.value = ''
        }}
      />
    </>
  )
}
