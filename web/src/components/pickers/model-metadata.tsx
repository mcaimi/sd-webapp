import { ExternalLink, Info } from 'lucide-react'
import { useState } from 'react'

import { useModelMetadata, useModels } from '@/api/queries'
import type { FluxLayout } from '@/api/types'
import { QueryError } from '@/components/query-error'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Skeleton } from '@/components/ui/skeleton'
import type { FamilyId, ResourceType } from '@/lib/families'
import { civitaiLabel, fluxComponents } from '@/lib/models'

import { ModelThumb } from './model-thumb'
import { NsfwBadge } from './nsfw-badge'

const PREVIEW_KEYS = 10

/** Info button showing a model's Civitai details and safetensors header metadata. Fetched on first open. */
export function ModelMetadataButton({
  family,
  resource,
  name,
}: {
  family: FamilyId
  resource: ResourceType
  name: string
}) {
  const [open, setOpen] = useState(false)
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button type="button" variant="ghost" size="icon-sm" aria-label={`Metadata for ${name}`}>
          <Info />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 space-y-3" align="end">
        {open && (
          <>
            <ModelDetails family={family} resource={resource} name={name} />
            <MetadataContent family={family} resource={resource} name={name} />
          </>
        )}
      </PopoverContent>
    </Popover>
  )
}

/** What the models list knows beyond the file: Civitai match, preview, Flux layout. */
function ModelDetails({ family, resource, name }: { family: FamilyId; resource: ResourceType; name: string }) {
  const { data } = useModels(family, resource)
  const model = data?.find((m) => m.name === name)
  if (!model) return null
  const info = model.civitai
  const label = civitaiLabel(model)
  const layout = model.flux_layout ? fluxComponents(model.flux_layout as Partial<FluxLayout>) : undefined
  if (!label && !model.has_preview && !layout) return null

  return (
    <div className="space-y-3 border-b pb-3">
      <div className="flex gap-3">
        {model.has_preview && (
          <ModelThumb family={family} resource={resource} model={model} className="size-16 rounded-md" />
        )}
        {label && (
          <div className="min-w-0 flex-1 space-y-1">
            <p className="text-sm font-medium">{label}</p>
            <div className="flex flex-wrap items-center gap-1">
              {info?.base_model && (
                <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
                  {info.base_model}
                </Badge>
              )}
              {info?.nsfw && <NsfwBadge />}
            </div>
            {info?.url && (
              <a
                href={info.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-xs font-medium underline underline-offset-4 hover:text-muted-foreground"
              >
                View on Civitai
                <ExternalLink className="size-3" aria-hidden />
              </a>
            )}
          </div>
        )}
      </div>
      {!!info?.trained_words.length && (
        <div className="text-xs">
          <p className="text-muted-foreground">Trigger words</p>
          <p className="break-words">{info.trained_words.join(', ')}</p>
        </div>
      )}
      {layout && (
        <div className="text-xs">
          <p className="text-muted-foreground">Bundled components</p>
          <p>{layout.bundled.join(', ') || 'None'}</p>
          {layout.shared.length > 0 && (
            <p className="text-muted-foreground">{layout.shared.join(', ')} from the shared Flux components</p>
          )}
        </div>
      )}
    </div>
  )
}

function MetadataContent({ family, resource, name }: { family: FamilyId; resource: ResourceType; name: string }) {
  const { data, error, isPending, refetch } = useModelMetadata(family, resource, name)
  const [showAll, setShowAll] = useState(false)

  if (isPending)
    return (
      <div className="space-y-2">
        <Skeleton className="h-4 w-2/3" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-1/2" />
      </div>
    )
  if (error) return <QueryError error={error} onRetry={() => refetch()} />

  const entries = Object.entries(data.metadata ?? {})
  const shown = showAll ? entries : entries.slice(0, PREVIEW_KEYS)
  return (
    <div className="space-y-3">
      <p className="truncate text-sm font-medium" title={data.path}>
        {data.name}
      </p>
      {entries.length === 0 ? (
        <p className="text-sm text-muted-foreground">No metadata in this file.</p>
      ) : (
        <dl className="max-h-72 space-y-2 overflow-y-auto text-xs">
          {shown.map(([key, value]) => (
            <div key={key}>
              <dt className="text-muted-foreground">{key}</dt>
              <dd className="line-clamp-3 break-all">{typeof value === 'string' ? value : JSON.stringify(value)}</dd>
            </div>
          ))}
        </dl>
      )}
      {entries.length > PREVIEW_KEYS && (
        <Button variant="link" size="xs" className="px-0" onClick={() => setShowAll((s) => !s)}>
          {showAll ? 'Show less' : `Show all ${entries.length} keys`}
        </Button>
      )}
    </div>
  )
}
