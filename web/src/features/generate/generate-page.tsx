import { useMutation } from '@tanstack/react-query'
import { Loader2, Settings2, Sparkles } from 'lucide-react'
import { useEffect } from 'react'
import { toast } from 'sonner'

import { useModels } from '@/api/queries'
import { PromptField } from '@/components/form/prompt-field'
import { ImageResults } from '@/components/jobs/image-results'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Kbd } from '@/components/ui/kbd'
import { Label } from '@/components/ui/label'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useTrackedJobs, type TrackedJob } from '@/jobs/job-store'
import { submitCompare, submitGenerate } from '@/jobs/submit'
import { useDraft } from '@/lib/drafts'
import { FAMILIES, type FamilyId } from '@/lib/families'
import { paramsFromRequest } from '@/lib/generation'
import { patchGenerationDraft, useGenerationDraft } from '@/lib/generation-draft'
import { MOD_KEY } from '@/lib/platform'
import { modelDisplayName } from '@/lib/models'

import { GenerationSettings } from './generation-settings'

type Mode = 'single' | 'compare'
const MAX_COMPARE = 6

export function GeneratePage({ family }: { family: FamilyId }) {
  const [params, setParams] = useGenerationDraft('generate', family)
  const [ui, setUi] = useDraft(`generate-ui:${family}`, () => ({
    mode: 'single' as Mode,
    compareModels: [] as string[],
  }))
  const { data: checkpoints } = useModels(family, 'checkpoints')
  const info = FAMILIES[family]

  // Default to the first checkpoint, and recover if the saved one disappeared.
  useEffect(() => {
    if (checkpoints?.length && !checkpoints.some((m) => m.name === params.model)) {
      setParams({ model: checkpoints[0].name })
    }
  }, [checkpoints, params.model, setParams])

  const submit = useMutation({
    mutationFn: () =>
      ui.mode === 'compare' ? submitCompare(family, params, ui.compareModels) : submitGenerate(family, params),
    onError: (error) => toast.error('Could not start generation', { description: error.message }),
  })

  const problem = !params.prompt.trim()
    ? 'Describe the image first'
    : ui.mode === 'single' && !params.model
      ? 'No checkpoint selected'
      : ui.mode === 'compare' && ui.compareModels.length < 2
        ? 'Pick at least two checkpoints'
        : null
  const trySubmit = () => {
    if (problem) toast.info(problem)
    else if (!submit.isPending) submit.mutate()
  }

  const allJobs = useTrackedJobs()
  const jobs = allJobs.filter((j) => j.family === family && (j.kind === 'generate' || j.kind === 'compare'))
  const useSettings = (job: TrackedJob) => {
    patchGenerationDraft('generate', family, paramsFromRequest(job.request))
    toast.success('Settings loaded into the form')
  }

  const showNegative = info.negativePrompt || params.trueCfgScale > 1
  const submitLabel =
    ui.mode === 'compare'
      ? `Compare ${ui.compareModels.length || ''} models`.replace('  ', ' ')
      : params.count > 1
        ? `Generate ${params.count}`
        : 'Generate'

  const settings = (
    <GenerationSettings
      family={family}
      params={params}
      onChange={setParams}
      showModel={ui.mode === 'single'}
      showAdapters={ui.mode === 'single'}
      showBatch={ui.mode === 'single'}
    />
  )

  const submitButton = (
    <Button className="flex-1" size="lg" onClick={trySubmit} disabled={submit.isPending} aria-describedby="submit-hint">
      {submit.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />}
      {submitLabel}
    </Button>
  )

  return (
    <div className="grid flex-1 gap-6 pb-24 lg:grid-cols-[minmax(20rem,24rem)_1fr] lg:pb-0">
      <aside className="space-y-5 lg:sticky lg:top-20 lg:max-h-[calc(100svh-6.5rem)] lg:self-start lg:overflow-y-auto lg:pr-2">
        {/* A mode switch rather than tabs: both modes share the same form below. */}
        <ToggleGroup
          type="single"
          variant="outline"
          aria-label="Mode"
          value={ui.mode}
          onValueChange={(mode) => mode && setUi({ mode: mode as Mode })}
          className="w-full"
        >
          <ToggleGroupItem value="single" className="flex-1">
            Generate
          </ToggleGroupItem>
          <ToggleGroupItem value="compare" className="flex-1">
            Compare models
          </ToggleGroupItem>
        </ToggleGroup>

        <PromptField
          label="Prompt"
          value={params.prompt}
          onChange={(prompt) => setParams({ prompt })}
          placeholder={`What should ${info.shortLabel} draw?`}
          onSubmit={trySubmit}
          autoFocus
        />
        {showNegative ? (
          <PromptField
            label="Negative prompt"
            value={params.negativePrompt}
            onChange={(negativePrompt) => setParams({ negativePrompt })}
            placeholder="What to avoid"
            rows={2}
            onSubmit={trySubmit}
          />
        ) : (
          <p className="text-xs text-muted-foreground">
            {info.shortLabel} ignores negative prompts unless True CFG is above 1.
          </p>
        )}

        {ui.mode === 'compare' && (
          <CompareModelPicker
            family={family}
            value={ui.compareModels}
            onChange={(compareModels) => setUi({ compareModels })}
          />
        )}

        <div className="hidden lg:block">{settings}</div>

        <div className="sticky bottom-0 hidden space-y-1.5 bg-background pt-2 pb-1 lg:block">
          <div className="flex">{submitButton}</div>
          <p id="submit-hint" className="text-center text-xs text-muted-foreground">
            <Kbd>{MOD_KEY}</Kbd> + <Kbd>Enter</Kbd> in a prompt to submit
          </p>
        </div>
      </aside>

      <section aria-label="Results" className="min-w-0">
        <ImageResults jobs={jobs} onUseSettings={useSettings} emptyState={<EmptyResults family={family} />} />
      </section>

      {/* Mobile: settings in a bottom sheet, submit always reachable. */}
      <div className="fixed inset-x-0 bottom-0 z-20 flex gap-2 border-t bg-background/90 p-3 backdrop-blur-md lg:hidden">
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="outline" size="lg" aria-label="Generation settings">
              <Settings2 />
              Settings
            </Button>
          </SheetTrigger>
          <SheetContent side="bottom" className="max-h-[85svh] gap-0 rounded-t-2xl p-0">
            <SheetHeader className="border-b">
              <SheetTitle>Settings · {info.shortLabel}</SheetTitle>
            </SheetHeader>
            <ScrollArea className="max-h-[calc(85svh-4rem)]">
              <div className="p-4">{settings}</div>
            </ScrollArea>
          </SheetContent>
        </Sheet>
        {submitButton}
      </div>
    </div>
  )
}

function CompareModelPicker({
  family,
  value,
  onChange,
}: {
  family: FamilyId
  value: string[]
  onChange: (models: string[]) => void
}) {
  const { data: checkpoints = [] } = useModels(family, 'checkpoints')
  const toggle = (name: string, on: boolean) => onChange(on ? [...value, name] : value.filter((m) => m !== name))

  return (
    <fieldset className="space-y-2">
      <legend className="text-sm font-medium">
        Checkpoints{' '}
        <span className="font-normal text-muted-foreground">
          {value.length}/{MAX_COMPARE}
        </span>
      </legend>
      <p className="text-xs text-muted-foreground">
        Same prompt and seed on each. Comparisons use checkpoints as-is (no LoRAs or custom VAE).
      </p>
      <div className="max-h-56 space-y-1 overflow-y-auto rounded-lg border p-2">
        {checkpoints.length === 0 && <p className="p-1 text-sm text-muted-foreground">No checkpoints installed.</p>}
        {checkpoints.map((m) => {
          const checked = value.includes(m.name)
          return (
            <Label
              key={m.name}
              className="flex cursor-pointer items-center gap-2 rounded-md p-1.5 font-normal hover:bg-accent"
            >
              <Checkbox
                checked={checked}
                disabled={!checked && value.length >= MAX_COMPARE}
                onCheckedChange={(on) => toggle(m.name, on === true)}
              />
              <span className="truncate">{modelDisplayName(m.name)}</span>
            </Label>
          )
        })}
      </div>
    </fieldset>
  )
}

function EmptyResults({ family }: { family: FamilyId }) {
  return (
    <div className="flex h-full min-h-80 flex-col items-center justify-center gap-3 rounded-xl border border-dashed p-8 text-center">
      <div className="flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Sparkles className="size-5" aria-hidden />
      </div>
      <h2 className="font-medium">Your {FAMILIES[family].shortLabel} images will appear here</h2>
      <p className="max-w-sm text-sm text-muted-foreground">
        Write a prompt and generate. Jobs keep running while you browse, and you can follow them from the Jobs panel.
      </p>
    </div>
  )
}
