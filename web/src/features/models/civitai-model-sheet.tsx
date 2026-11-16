import { Check, Copy, ExternalLink, Star, ThumbsUp, Download as DownloadIcon } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'

import type { CivitaiModelSummary, CivitaiVersion } from '@/api/types'
import { NsfwBadge } from '@/components/pickers/nsfw-badge'
import { QueryError } from '@/components/query-error'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import { civitaiModelUrl, htmlToText } from '@/lib/civitai'
import { FAMILIES, type FamilyId } from '@/lib/families'
import { formatCount } from '@/lib/format'

import { CivitaiFileRow } from './civitai-file-row'
import { CivitaiPreview } from './civitai-preview'
import { useCivitaiModel } from './civitai-queries'

interface CivitaiModelSheetProps {
  family: FamilyId
  /** The search result to show; null closes the sheet. */
  summary: CivitaiModelSummary | null
  nsfw: boolean
  onOpenChange: (open: boolean) => void
}

export function CivitaiModelSheet({ family, summary, nsfw, onOpenChange }: CivitaiModelSheetProps) {
  return (
    <Sheet open={!!summary} onOpenChange={onOpenChange}>
      <SheetContent className="flex w-full flex-col gap-0 p-0 sm:max-w-xl">
        {summary && <ModelDetail key={summary.id} family={family} summary={summary} nsfw={nsfw} />}
      </SheetContent>
    </Sheet>
  )
}

function ModelDetail({ family, summary, nsfw }: { family: FamilyId; summary: CivitaiModelSummary; nsfw: boolean }) {
  const { data: model, error, isPending, refetch } = useCivitaiModel(summary.id, family, nsfw)
  const [versionId, setVersionId] = useState<number>()
  const versions = model?.versions ?? []
  const version = versions.find((v) => v.id === versionId) ?? versions[0]
  const stats = (model ?? summary).stats

  return (
    <>
      <SheetHeader className="border-b pr-12">
        <SheetTitle className="leading-snug">{summary.name}</SheetTitle>
        <SheetDescription className="flex flex-wrap items-center gap-x-3 gap-y-1">
          {summary.creator && <span>by {summary.creator}</span>}
          <span className="inline-flex items-center gap-1 tabular-nums">
            <DownloadIcon className="size-3" aria-hidden />
            {formatCount(stats.downloads ?? 0)} downloads
          </span>
          {!!stats.thumbs_up && (
            <span className="inline-flex items-center gap-1 tabular-nums">
              <ThumbsUp className="size-3" aria-hidden />
              {formatCount(stats.thumbs_up)}
            </span>
          )}
          {stats.rating != null && stats.rating > 0 && (
            <span className="inline-flex items-center gap-1 tabular-nums">
              <Star className="size-3" aria-hidden />
              {stats.rating.toFixed(1)} ({formatCount(stats.rating_count ?? 0)})
            </span>
          )}
        </SheetDescription>
      </SheetHeader>
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-4">
        <div className="flex gap-4">
          <CivitaiPreview
            preview={version?.preview ?? summary.preview}
            width={600}
            revealable
            className="aspect-[3/4] w-36 shrink-0 rounded-lg sm:w-44"
          />
          <div className="min-w-0 flex-1 space-y-3">
            <div className="flex flex-wrap gap-1">
              {summary.type && <Badge variant="secondary">{summary.type}</Badge>}
              {(model ?? summary).base_models.map((base) => (
                <Badge key={base} variant="outline">
                  {base}
                </Badge>
              ))}
              {summary.nsfw && <NsfwBadge className="h-5 px-1.5" />}
            </div>
            {!!summary.tags.length && (
              <p className="line-clamp-3 text-xs text-muted-foreground">{summary.tags.join(' · ')}</p>
            )}
            <a
              href={civitaiModelUrl(summary.id, version?.id)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-sm font-medium underline underline-offset-4 hover:text-muted-foreground"
            >
              View on Civitai
              <ExternalLink className="size-3.5" aria-hidden />
            </a>
          </div>
        </div>

        {error ? (
          <QueryError error={error} onRetry={() => refetch()} title="Could not load this model" />
        ) : isPending ? (
          <div className="space-y-3" aria-busy>
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
          </div>
        ) : !version ? (
          <p className="text-sm text-muted-foreground">This model has no versions to download.</p>
        ) : (
          <>
            {versions.length > 1 && (
              <div className="space-y-2">
                <Label htmlFor="civitai-version">Version</Label>
                <Select value={String(version.id)} onValueChange={(id) => setVersionId(Number(id))}>
                  <SelectTrigger id="civitai-version" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {versions.map((v) => (
                      <SelectItem key={v.id} value={String(v.id)}>
                        {v.name}
                        {v.base_model && <span className="text-muted-foreground"> · {v.base_model}</span>}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            <VersionDetail family={family} version={version} modelName={summary.name} />
            {model.description && <Description text={htmlToText(model.description)} />}
          </>
        )}
      </div>
    </>
  )
}

function VersionDetail({
  family,
  version,
  modelName,
}: {
  family: FamilyId
  version: CivitaiVersion
  modelName: string
}) {
  const otherFamily = version.model_type && version.model_type !== family ? version.model_type : undefined
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        {version.base_model && <span>Base model: {version.base_model}</span>}
        {version.published_at && <span>· Published {new Date(version.published_at).toLocaleDateString()}</span>}
        {version.early_access && <Badge variant="outline">Early access</Badge>}
      </div>
      {otherFamily && (
        <p className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-2.5 text-xs">
          This version is made for {FAMILIES[otherFamily as FamilyId]?.label ?? otherFamily}, not{' '}
          {FAMILIES[family].label}.
        </p>
      )}
      {!!version.trained_words.length && <TriggerWords words={version.trained_words} />}
      <section className="space-y-2">
        <h3 className="text-xs font-medium text-muted-foreground uppercase">Files</h3>
        {version.files.length === 0 ? (
          <p className="text-sm text-muted-foreground">No files in this version.</p>
        ) : (
          <ul className="space-y-2">
            {version.files.map((file) => (
              <CivitaiFileRow key={file.id} family={family} version={version} file={file} modelName={modelName} />
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}

function TriggerWords({ words }: { words: string[] }) {
  const [copied, setCopied] = useState<string>()
  const copy = (word: string) =>
    navigator.clipboard.writeText(word).then(
      () => setCopied(word),
      () => toast.error('Could not copy to the clipboard'),
    )
  return (
    <section className="space-y-1.5">
      <h3 className="text-xs font-medium text-muted-foreground uppercase">Trigger words</h3>
      <ul className="flex flex-wrap gap-1">
        {words.map((word) => (
          <li key={word}>
            <button
              type="button"
              onClick={() => copy(word)}
              title={`Copy "${word}"`}
              className="inline-flex max-w-full items-center gap-1 rounded-sm border px-1.5 py-0.5 text-xs hover:bg-muted"
            >
              <span className="truncate">{word}</span>
              {copied === word ? <Check className="size-3" aria-hidden /> : <Copy className="size-3" aria-hidden />}
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}

function Description({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false)
  if (!text) return null
  return (
    <section className="space-y-1.5">
      <h3 className="text-xs font-medium text-muted-foreground uppercase">Description</h3>
      <p className={expanded ? 'text-sm whitespace-pre-line' : 'line-clamp-6 text-sm whitespace-pre-line'}>{text}</p>
      {text.length > 400 && (
        <Button variant="link" size="xs" className="px-0" onClick={() => setExpanded((e) => !e)}>
          {expanded ? 'Show less' : 'Show more'}
        </Button>
      )}
    </section>
  )
}
