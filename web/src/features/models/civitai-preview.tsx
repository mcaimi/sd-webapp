import { EyeOff, ImageOff } from 'lucide-react'
import { useState } from 'react'

import { useCivitaiStatus } from '@/api/queries'
import type { CivitaiPreview as Preview } from '@/api/types'
import { civitaiThumbUrl, isNsfwPreview } from '@/lib/civitai'
import { cn } from '@/lib/utils'

const prefersReducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false

/**
 * A Civitai preview image or video. NSFW previews are blurred unless the
 * backend shows NSFW content by default; `revealable` lets the user unblur one.
 */
export function CivitaiPreview({
  preview,
  width,
  revealable,
  className,
}: {
  preview: Preview | null | undefined
  /** Rendition width to request from Civitai. */
  width?: number
  revealable?: boolean
  className?: string
}) {
  const { data: status } = useCivitaiStatus()
  const [revealed, setRevealed] = useState(false)
  const [failed, setFailed] = useState(false)
  const blurred = isNsfwPreview(preview) && !status?.nsfw_default && !revealed

  if (!preview || failed) {
    return (
      <div className={cn('flex items-center justify-center bg-muted text-muted-foreground', className)}>
        <ImageOff className="size-6 opacity-60" aria-hidden />
      </div>
    )
  }

  const src = civitaiThumbUrl(preview.url, width)
  const media = cn('size-full object-cover transition-[filter,scale] duration-300', blurred && 'scale-110 blur-xl')
  return (
    <div className={cn('relative overflow-hidden bg-muted', className)}>
      {preview.type === 'video' ? (
        <video
          src={src}
          className={media}
          muted
          loop
          playsInline
          autoPlay={!prefersReducedMotion()}
          onError={() => setFailed(true)}
          aria-hidden
        />
      ) : (
        <img src={src} alt="" loading="lazy" className={media} onError={() => setFailed(true)} />
      )}
      {blurred &&
        (revealable ? (
          <button
            type="button"
            onClick={() => setRevealed(true)}
            className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-black/20 text-xs font-medium text-white"
          >
            <EyeOff className="size-4" aria-hidden />
            NSFW preview · Show
          </button>
        ) : (
          <span className="absolute inset-0 flex items-center justify-center" aria-hidden>
            <span className="flex items-center gap-1 rounded-sm bg-black/50 px-1.5 py-0.5 text-[10px] font-medium text-white">
              <EyeOff className="size-3" />
              NSFW
            </span>
          </span>
        ))}
    </div>
  )
}
