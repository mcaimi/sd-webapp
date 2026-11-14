import { useQueryClient } from '@tanstack/react-query'
import { Copy, Download, Expand, Images, Shuffle, Trash2 } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'

import type { OutputItem } from '@/api/schema'
import { DetailList, Lightbox, type LightboxItem } from '@/components/lightbox'
import { LoadMore } from '@/components/load-more'
import { PageHeader } from '@/components/page-header'
import { QueryError } from '@/components/query-error'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { download } from '@/lib/download'
import { FAMILIES } from '@/lib/families'
import { formatBytes, formatRelative } from '@/lib/format'
import { paramsFromOutputMetadata, type OutputMetadata } from '@/lib/generation'
import { patchGenerationDraft } from '@/lib/generation-draft'
import { modelDisplayName } from '@/lib/models'
import { useActiveFamily } from '@/lib/use-active-family'

import { outputMetadataQuery, scheduleDelete, useOutputMetadata, useOutputs, usePendingDeletes } from './outputs'

const GRID = 'grid grid-cols-2 gap-2 sm:grid-cols-3 sm:gap-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6'

export function GalleryPage() {
  const { data, error, isPending, refetch, fetchNextPage, hasNextPage, isFetchingNextPage } = useOutputs()
  const pendingDeletes = usePendingDeletes()
  const [openId, setOpenId] = useState<string | null>(null)
  const actions = useOutputActions()

  // Pages are offset-based, so new generations can shift an item onto the next page: dedupe.
  const items = useMemo(() => {
    const seen = new Set<string>()
    return (data?.pages ?? [])
      .flatMap((page) => page.items)
      .filter((item) => !seen.has(item.id) && seen.add(item.id) && !pendingDeletes.has(item.id))
  }, [data, pendingDeletes])
  const total = data ? Math.max(0, data.pages[0].total - pendingDeletes.size) : undefined

  const openIndex = openId ? items.findIndex((i) => i.id === openId) : -1
  // Deleting the open image moves the viewer to its neighbour instead of closing it.
  const lastOpenIndex = useRef(-1)
  useEffect(() => {
    if (openIndex >= 0) lastOpenIndex.current = openIndex
  }, [openIndex])
  useEffect(() => {
    if (openId && openIndex < 0 && pendingDeletes.has(openId)) {
      const next = items[Math.min(lastOpenIndex.current, items.length - 1)]
      setOpenId(next?.id ?? null)
    }
  }, [openId, openIndex, pendingDeletes, items])
  const openItem = openIndex >= 0 ? items[openIndex] : undefined
  const { data: openMeta } = useOutputMetadata(openItem?.id)

  const lightboxItems: LightboxItem[] = items.map((item) =>
    toLightboxItem(item, item.id === openItem?.id ? openMeta : undefined, actions),
  )

  const header = (
    <PageHeader
      title="Gallery"
      description={
        total === undefined
          ? 'Generations saved by the backend.'
          : `${total} ${total === 1 ? 'image' : 'images'} saved by the backend.`
      }
    />
  )

  if (error) {
    return (
      <>
        {header}
        <QueryError error={error} onRetry={() => refetch()} title="Could not load the gallery" />
      </>
    )
  }

  if (isPending) {
    return (
      <>
        {header}
        <div className={GRID} aria-busy>
          {Array.from({ length: 12 }, (_, i) => (
            <Skeleton key={i} className="aspect-square rounded-xl" />
          ))}
        </div>
      </>
    )
  }

  if (items.length === 0) {
    return (
      <>
        {header}
        <div className="flex flex-1 flex-col items-center justify-center gap-3 rounded-xl border border-dashed p-10 text-center">
          <div className="flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
            <Images className="size-5" aria-hidden />
          </div>
          <h2 className="font-medium">No saved images yet</h2>
          <p className="max-w-sm text-sm text-muted-foreground">
            Images you generate are saved by the backend and collected here.
          </p>
        </div>
      </>
    )
  }

  return (
    <>
      {header}
      <ul className={GRID} aria-label="Saved images">
        <AnimatePresence initial={false}>
          {items.map((item, index) => (
            <GalleryTile
              key={item.id}
              item={item}
              // Stagger within a page so appended pages animate in too.
              delay={(index % 24) * 0.02}
              onOpen={() => setOpenId(item.id)}
              actions={actions}
            />
          ))}
        </AnimatePresence>
      </ul>
      <LoadMore hasNextPage={hasNextPage} isFetching={isFetchingNextPage} onLoadMore={() => fetchNextPage()} />
      <Lightbox
        items={lightboxItems}
        index={openIndex >= 0 ? openIndex : null}
        onIndexChange={(i) => setOpenId(i === null ? null : (items[i]?.id ?? null))}
      />
    </>
  )
}

interface OutputActions {
  download: (item: OutputItem) => void
  remix: (item: OutputItem) => void
  copyPrompt: (item: OutputItem) => void
  remove: (item: OutputItem) => void
}

function useOutputActions(): OutputActions {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [family] = useActiveFamily()
  const metadata = (item: OutputItem) => queryClient.fetchQuery(outputMetadataQuery(item.id))
  const fail = (title: string) => (error: Error) => toast.error(title, { description: error.message })

  return {
    download: (item) => {
      const seed = queryClient.getQueryData<OutputMetadata>(outputMetadataQuery(item.id).queryKey)?.generation
        ?.instances?.[0]?.seed
      download(item.image_url, `output_${seed ?? item.id.slice(0, 8)}.png`)
    },
    remix: (item) =>
      metadata(item)
        .then((meta) => {
          patchGenerationDraft('generate', family, paramsFromOutputMetadata(meta))
          navigate(`/generate/${family}`)
          toast.success('Settings loaded into Generate', {
            description: `Saved images don't record their checkpoint, so the current ${FAMILIES[family].shortLabel} checkpoint is kept.`,
          })
        })
        .catch(fail('Could not load settings')),
    copyPrompt: (item) =>
      metadata(item)
        .then((meta) => navigator.clipboard.writeText(meta.generation?.instances?.[0]?.prompt ?? ''))
        .then(() => toast.success('Prompt copied'))
        .catch(fail('Could not copy prompt')),
    remove: scheduleDelete,
  }
}

function GalleryTile({
  item,
  delay,
  onOpen,
  actions,
}: {
  item: OutputItem
  delay: number
  onOpen: () => void
  actions: OutputActions
}) {
  const [loaded, setLoaded] = useState(false)
  return (
    <motion.li
      layout
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0, transition: { delay, duration: 0.25 } }}
      exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
      className="group/tile relative aspect-square overflow-hidden rounded-xl bg-muted"
    >
      <button
        type="button"
        onClick={onOpen}
        className="block size-full cursor-zoom-in"
        aria-label={`Open image from ${formatRelative(item.created_at)}`}
      >
        <motion.img
          layoutId={`output-${item.id}`}
          src={item.image_url}
          alt=""
          loading="lazy"
          decoding="async"
          onLoad={() => setLoaded(true)}
          className="size-full object-cover transition-[opacity,scale] duration-500 group-hover/tile:scale-105"
          style={{ opacity: loaded ? 1 : 0 }}
        />
      </button>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-1 bg-gradient-to-t from-black/70 to-transparent p-2 pt-8 opacity-0 transition-opacity group-focus-within/tile:opacity-100 group-hover/tile:opacity-100 [@media(hover:none)]:opacity-100">
        <span className="truncate text-xs text-white/90">{formatRelative(item.created_at)}</span>
        <div className="pointer-events-auto flex shrink-0 gap-1">
          <TileAction label="Open" onClick={onOpen}>
            <Expand />
          </TileAction>
          <TileAction label="Remix in Generate" onClick={() => actions.remix(item)}>
            <Shuffle />
          </TileAction>
          <TileAction label="Download" onClick={() => actions.download(item)}>
            <Download />
          </TileAction>
          <TileAction label="Delete" onClick={() => actions.remove(item)}>
            <Trash2 />
          </TileAction>
        </div>
      </div>
    </motion.li>
  )
}

function TileAction({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="secondary"
          size="icon-sm"
          className="bg-black/50 text-white backdrop-blur hover:bg-black/70"
          onClick={onClick}
          aria-label={label}
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

/** Infinite scroll via a sentinel, with a button for when the observer can't fire. */

function toLightboxItem(item: OutputItem, meta: OutputMetadata | undefined, actions: OutputActions): LightboxItem {
  const instance = meta?.generation?.instances?.[0]
  const loras = Object.values(meta?.loras ?? {})
  return {
    key: item.id,
    src: item.image_url,
    alt: instance?.prompt || 'Saved image',
    layoutId: `output-${item.id}`,
    title: instance ? <span className="line-clamp-4">{instance.prompt}</span> : <Skeleton className="h-5 w-3/4" />,
    actions: (
      <>
        <Button size="sm" onClick={() => actions.download(item)}>
          <Download />
          Download
        </Button>
        <Button size="sm" variant="outline" onClick={() => actions.remix(item)}>
          <Shuffle />
          Remix
        </Button>
        <Button size="sm" variant="outline" onClick={() => actions.copyPrompt(item)}>
          <Copy />
          Prompt
        </Button>
        <Button size="sm" variant="ghost" onClick={() => actions.remove(item)} aria-label="Delete image">
          <Trash2 />
        </Button>
      </>
    ),
    details: (
      <DetailList
        entries={[
          ['Negative', instance?.negative_prompt],
          ['Size', instance && `${instance.width}×${instance.height}`],
          ['Steps', instance?.num_inference_steps],
          ['CFG', instance?.guidance_scale],
          ['Scheduler', instance?.scheduler],
          ['Seed', instance?.seed],
          ['LoRAs', loras.map((l) => `${modelDisplayName(l.name)} (${l.merge_strength})`).join(', ')],
          ['VAE', meta?.vae && modelDisplayName(meta.vae)],
          ['Created', formatRelative(item.created_at)],
          ['File size', formatBytes(item.size_bytes)],
        ]}
      />
    ),
  }
}
