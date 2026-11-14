import { Boxes } from 'lucide-react'
import { useState } from 'react'

import { useCivitaiStatus } from '@/api/queries'
import type { ModelEntry } from '@/api/types'
import type { FamilyId, ResourceType } from '@/lib/families'
import { modelPreviewUrl } from '@/lib/models'
import { cn } from '@/lib/utils'

/**
 * A model's Civitai preview, or a placeholder. NSFW previews stay blurred
 * unless the backend shows NSFW content by default.
 */
export function ModelThumb({
  family,
  resource,
  model,
  className,
}: {
  family: FamilyId
  resource: ResourceType
  model: ModelEntry
  className?: string
}) {
  const [failed, setFailed] = useState(false)
  const { data: civitai } = useCivitaiStatus()
  const blur = !!model.civitai?.nsfw && !civitai?.nsfw_default

  return (
    <div
      className={cn('flex shrink-0 items-center justify-center overflow-hidden rounded-sm bg-muted', className)}
      aria-hidden
    >
      {model.has_preview && !failed ? (
        <img
          src={modelPreviewUrl(family, resource, model.name)}
          alt=""
          loading="lazy"
          onError={() => setFailed(true)}
          className={cn('size-full object-cover', blur && 'scale-110 blur-md')}
        />
      ) : (
        <Boxes className="size-1/2 text-muted-foreground/60" />
      )}
    </div>
  )
}
