import { Download, Globe, KeyRound, Search, Star, ThumbsUp } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import type { CivitaiModelSummary, CivitaiStatusResponse } from '@/api/types'
import { LoadMore } from '@/components/load-more'
import { NsfwBadge } from '@/components/pickers/nsfw-badge'
import { QueryError } from '@/components/query-error'
import { Badge } from '@/components/ui/badge'
import { InputGroup, InputGroupAddon, InputGroupInput } from '@/components/ui/input-group'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import type { FamilyId, ResourceType } from '@/lib/families'
import { formatCount } from '@/lib/format'
import { cn } from '@/lib/utils'

import { CivitaiModelSheet } from './civitai-model-sheet'
import { CivitaiPreview } from './civitai-preview'
import { useCivitaiSearch } from './civitai-queries'
import { EmptyState } from './empty-state'

const SEARCH_DEBOUNCE_MS = 400
const GRID = 'grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6'

/** Human labels for the backend's period options. */
const PERIOD_LABELS: Record<string, string> = {
  AllTime: 'All time',
  Year: 'This year',
  Month: 'This month',
  Week: 'This week',
  Day: 'Today',
}

export interface BrowseFilters {
  sort?: string
  period?: string
  nsfw?: boolean
}

interface CivitaiBrowserProps {
  family: FamilyId
  resource: ResourceType
  status: CivitaiStatusResponse
  filters: BrowseFilters
  onFiltersChange: (patch: BrowseFilters) => void
}

export function CivitaiBrowser({ family, resource, status, filters, onFiltersChange }: CivitaiBrowserProps) {
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [openModel, setOpenModel] = useState<CivitaiModelSummary | null>(null)
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query), SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [query])

  const sort = filters.sort && status.sort_options.includes(filters.sort) ? filters.sort : status.sort_options[0]
  const period =
    filters.period && status.period_options.includes(filters.period) ? filters.period : status.period_options[0]
  const nsfw = filters.nsfw ?? status.nsfw_default
  const search = useCivitaiSearch({ family, resource, query: debouncedQuery, sort, period, nsfw }, status.enabled)

  // Cursor pages can overlap when results shift between requests: dedupe.
  const items = useMemo(() => {
    const seen = new Set<number>()
    return (search.data?.pages ?? []).flatMap((p) => p.items).filter((m) => !seen.has(m.id) && seen.add(m.id))
  }, [search.data])

  return (
    <div className="space-y-4">
      {!status.has_api_key && (
        <p className="flex items-start gap-2 rounded-lg border bg-muted/40 p-3 text-sm text-muted-foreground">
          <KeyRound className="mt-0.5 size-4 shrink-0" aria-hidden />
          The backend has no Civitai API key, so some downloads (most Flux models, early access files) and Flux
          component checks will fail.
        </p>
      )}
      <div className="flex flex-col gap-3 md:flex-row md:items-center">
        <InputGroup className="md:max-w-sm">
          <InputGroupAddon>
            <Search aria-hidden />
          </InputGroupAddon>
          <InputGroupInput
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search Civitai…"
            aria-label="Search Civitai"
            maxLength={200}
          />
        </InputGroup>
        <div className="flex flex-wrap items-center gap-3">
          <Select value={sort} onValueChange={(value) => onFiltersChange({ sort: value })}>
            <SelectTrigger className="w-40" aria-label="Sort">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {status.sort_options.map((option) => (
                <SelectItem key={option} value={option}>
                  {option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={period} onValueChange={(value) => onFiltersChange({ period: value })}>
            <SelectTrigger className="w-36" aria-label="Period">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {status.period_options.map((option) => (
                <SelectItem key={option} value={option}>
                  {PERIOD_LABELS[option] ?? option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="flex items-center gap-2">
            <Switch id="civitai-nsfw" checked={nsfw} onCheckedChange={(on) => onFiltersChange({ nsfw: on })} />
            <Label htmlFor="civitai-nsfw" className="font-normal">
              Include NSFW
            </Label>
          </div>
        </div>
      </div>

      {search.error ? (
        <QueryError error={search.error} onRetry={() => search.refetch()} title="Could not search Civitai" />
      ) : search.isPending ? (
        <div className={GRID} aria-busy>
          {Array.from({ length: 8 }, (_, i) => (
            <Skeleton key={i} className="aspect-[4/5] rounded-xl" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <EmptyState
          icon={Globe}
          title="No models found"
          description={`Nothing on Civitai matches these filters for ${resource} of this family.`}
        />
      ) : (
        <>
          <ul
            className={cn(GRID, search.isPlaceholderData && 'opacity-60 transition-opacity')}
            aria-label="Civitai models"
            aria-busy={search.isPlaceholderData}
          >
            {items.map((model) => (
              <li key={model.id}>
                <ModelCard model={model} onOpen={() => setOpenModel(model)} />
              </li>
            ))}
          </ul>
          <LoadMore
            hasNextPage={search.hasNextPage}
            isFetching={search.isFetchingNextPage}
            onLoadMore={() => search.fetchNextPage()}
          />
        </>
      )}

      <CivitaiModelSheet
        family={family}
        summary={openModel}
        nsfw={nsfw}
        onOpenChange={(open) => !open && setOpenModel(null)}
      />
    </div>
  )
}

function ModelCard({ model, onOpen }: { model: CivitaiModelSummary; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="group/card flex w-full flex-col overflow-hidden rounded-xl border bg-card text-left transition-shadow hover:shadow-md focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
    >
      <CivitaiPreview preview={model.preview} className="aspect-[4/5] w-full" />
      <div className="min-w-0 space-y-1.5 p-2.5">
        <p className="line-clamp-2 text-sm leading-snug font-medium">{model.name}</p>
        {model.creator && <p className="truncate text-xs text-muted-foreground">by {model.creator}</p>}
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-muted-foreground tabular-nums">
          <span className="inline-flex items-center gap-1" title="Downloads">
            <Download className="size-3" aria-hidden />
            <span className="sr-only">Downloads:</span>
            {formatCount(model.stats.downloads ?? 0)}
          </span>
          {!!model.stats.thumbs_up && (
            <span className="inline-flex items-center gap-1" title="Thumbs up">
              <ThumbsUp className="size-3" aria-hidden />
              <span className="sr-only">Thumbs up:</span>
              {formatCount(model.stats.thumbs_up)}
            </span>
          )}
          {model.stats.rating != null && model.stats.rating > 0 && (
            <span className="inline-flex items-center gap-1" title="Rating">
              <Star className="size-3" aria-hidden />
              <span className="sr-only">Rating:</span>
              {model.stats.rating.toFixed(1)}
            </span>
          )}
        </div>
        <div className="flex flex-wrap gap-1">
          {model.base_models.slice(0, 2).map((base) => (
            <Badge key={base} variant="outline" className="h-4 px-1.5 text-[10px]">
              {base}
            </Badge>
          ))}
          {model.nsfw && <NsfwBadge />}
        </div>
      </div>
    </button>
  )
}
