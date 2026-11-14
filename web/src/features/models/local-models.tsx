import { ExternalLink, Fingerprint, HardDrive, Loader2, Search } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'

import { useCivitaiStatus, useModels } from '@/api/queries'
import type { FluxLayout, ModelEntry } from '@/api/types'
import { ModelMetadataButton } from '@/components/pickers/model-metadata'
import { ModelThumb } from '@/components/pickers/model-thumb'
import { NsfwBadge } from '@/components/pickers/nsfw-badge'
import { QueryError } from '@/components/query-error'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { InputGroup, InputGroupAddon, InputGroupInput } from '@/components/ui/input-group'
import { Skeleton } from '@/components/ui/skeleton'
import { RESOURCE_LABELS, type FamilyId, type ResourceType } from '@/lib/families'
import { civitaiLabel, fluxComponents, modelDisplayName } from '@/lib/models'

import { useIdentifyModel } from './civitai-queries'
import { EmptyState } from './empty-state'

/** Installed models of one resource type, with their Civitai details. */
export function LocalModels({
  family,
  resource,
  onBrowse,
}: {
  family: FamilyId
  resource: ResourceType
  /** Switch to browsing Civitai; omitted when the integration is off. */
  onBrowse?: () => void
}) {
  const { data, error, isPending, refetch } = useModels(family, resource)
  const [filter, setFilter] = useState('')

  if (error) return <QueryError error={error} onRetry={() => refetch()} title="Could not load models" />
  if (isPending) {
    return (
      <div className="space-y-2" aria-busy>
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="h-16 rounded-lg" />
        ))}
      </div>
    )
  }
  if (data.length === 0) {
    return (
      <EmptyState
        icon={HardDrive}
        title={`No ${RESOURCE_LABELS[resource]} installed`}
        description="Models placed in the backend's model directories, or downloaded from Civitai, show up here."
      >
        {onBrowse && (
          <Button variant="outline" onClick={onBrowse}>
            Browse Civitai
          </Button>
        )}
      </EmptyState>
    )
  }

  const needle = filter.trim().toLowerCase()
  const shown = needle ? data.filter((m) => `${m.name} ${civitaiLabel(m) ?? ''}`.toLowerCase().includes(needle)) : data
  return (
    <div className="space-y-3">
      <InputGroup className="sm:max-w-sm">
        <InputGroupAddon>
          <Search aria-hidden />
        </InputGroupAddon>
        <InputGroupInput
          type="search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={`Filter ${data.length} installed…`}
          aria-label="Filter installed models"
        />
      </InputGroup>
      {shown.length === 0 ? (
        <p className="text-sm text-muted-foreground">No installed models match “{filter}”.</p>
      ) : (
        <ul className="divide-y rounded-lg border" aria-label={`Installed ${RESOURCE_LABELS[resource]}`}>
          {shown.map((model) => (
            <LocalModelRow key={model.name} family={family} resource={resource} model={model} />
          ))}
        </ul>
      )}
    </div>
  )
}

function LocalModelRow({ family, resource, model }: { family: FamilyId; resource: ResourceType; model: ModelEntry }) {
  const { data: civitaiStatus } = useCivitaiStatus()
  const identify = useIdentifyModel(family, resource)
  const info = model.civitai
  const label = civitaiLabel(model)
  const layout = model.flux_layout ? fluxComponents(model.flux_layout as Partial<FluxLayout>) : undefined

  const runIdentify = () =>
    identify.mutate(model.name, {
      onSuccess: (result) => {
        if (result.matched) toast.success(`Found ${modelDisplayName(model.name)} on Civitai`)
        else toast.info('Not found on Civitai', { description: 'No Civitai file has the same hash.' })
      },
      onError: (e) => toast.error('Could not look up this model', { description: e.message }),
    })

  return (
    <li className="flex items-center gap-3 p-3">
      <ModelThumb family={family} resource={resource} model={model} className="size-12 rounded-md" />
      <div className="min-w-0 flex-1 space-y-1">
        <p className="truncate text-sm font-medium" title={model.path}>
          {modelDisplayName(model.name)}
        </p>
        <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
          {label ? <span className="truncate">{label}</span> : <span>Not linked to Civitai</span>}
          {info?.base_model && (
            <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
              {info.base_model}
            </Badge>
          )}
          {info?.nsfw && <NsfwBadge />}
          {!!info?.trained_words.length && (
            <span title={info.trained_words.join(', ')}>
              · {info.trained_words.length} trigger {info.trained_words.length === 1 ? 'word' : 'words'}
            </span>
          )}
          {layout && <span>· bundles {layout.bundled.join(', ') || 'nothing'}</span>}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-1">
        {info?.url ? (
          <Button variant="ghost" size="icon-sm" asChild>
            <a href={info.url} target="_blank" rel="noreferrer" aria-label={`View ${label} on Civitai`}>
              <ExternalLink />
            </a>
          </Button>
        ) : (
          civitaiStatus?.enabled && (
            <Button
              variant="outline"
              size="xs"
              onClick={runIdentify}
              disabled={identify.isPending}
              title="Hash the file and look it up on Civitai. Large checkpoints take a while to hash."
            >
              {identify.isPending ? <Loader2 className="animate-spin" /> : <Fingerprint />}
              {identify.isPending ? 'Identifying…' : 'Identify'}
            </Button>
          )
        )}
        <ModelMetadataButton family={family} resource={resource} name={model.name} />
      </div>
    </li>
  )
}
