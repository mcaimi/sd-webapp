import { Info } from 'lucide-react'
import { useState } from 'react'

import { useModelMetadata } from '@/api/queries'
import { QueryError } from '@/components/query-error'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Skeleton } from '@/components/ui/skeleton'
import type { FamilyId, ResourceType } from '@/lib/families'

const PREVIEW_KEYS = 10

/** Info button showing a model's safetensors header metadata. Fetched on first open. */
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
      <PopoverContent className="w-80" align="end">
        {open && <MetadataContent family={family} resource={resource} name={name} />}
      </PopoverContent>
    </Popover>
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
