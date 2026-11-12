import { useMutation } from '@tanstack/react-query'
import { Brush, Loader2, Settings2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'

import { useModels } from '@/api/queries'
import { PromptField } from '@/components/form/prompt-field'
import { SliderField } from '@/components/form/slider-field'
import { ImageResults } from '@/components/jobs/image-results'
import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { GenerationSettings } from '@/features/generate/generation-settings'
import { useTrackedJobs, type TrackedJob } from '@/jobs/job-store'
import { submitInpaint } from '@/jobs/submit'
import { useDraft } from '@/lib/drafts'
import { FAMILIES, type FamilyId } from '@/lib/families'
import { paramsFromRequest } from '@/lib/generation'
import { patchGenerationDraft, useGenerationDraft } from '@/lib/generation-draft'
import { MOD_KEY } from '@/lib/platform'

import { BeforeAfter } from './before-after'
import { ImageDropzone } from './image-dropzone'
import { maxSideFor } from './image-size'
import { MaskEditor, type MaskEditorHandle } from './mask-editor'
import { jobSources, workspaces, type SourceImage } from './source-images'

export function InpaintPage({ family }: { family: FamilyId }) {
  const info = FAMILIES[family]
  const [params, setParams] = useGenerationDraft('inpaint', family)
  const [ui, setUi] = useDraft(`inpaint-ui:${family}`, () => ({ strength: 0.85 }))
  const { data: checkpoints } = useModels(family, 'checkpoints')

  const [image, setImageState] = useState<SourceImage | undefined>(() => workspaces.get(family)?.image)
  const [hasMask, setHasMask] = useState(false)
  const editorRef = useRef<MaskEditorHandle>(null)

  const setImage = (next: SourceImage | undefined) => {
    if (next) workspaces.set(family, { image: next })
    else workspaces.delete(family)
    setHasMask(false)
    setImageState(next)
  }
  const persistMask = useCallback(
    (mask: ImageData, source: SourceImage) => {
      const workspace = workspaces.get(family)
      // Ignore masks from an image that has since been replaced.
      if (workspace?.image === source) workspace.mask = mask
    },
    [family],
  )

  useEffect(() => {
    if (checkpoints?.length && !checkpoints.some((m) => m.name === params.model)) {
      setParams({ model: checkpoints[0].name })
    }
  }, [checkpoints, params.model, setParams])

  const submit = useMutation({
    mutationFn: async () => {
      const mask = editorRef.current!.exportMask()
      const jobId = await submitInpaint(family, params, image!, mask, family === 'flux' ? ui.strength : undefined)
      jobSources.set(jobId, image!.url)
      return jobId
    },
    onError: (error) => toast.error('Could not start inpainting', { description: error.message }),
  })

  const problem = !image
    ? 'Choose an image first'
    : !hasMask
      ? 'Paint over the area to regenerate'
      : !params.prompt.trim()
        ? 'Describe what should appear in the masked area'
        : !params.model
          ? 'No checkpoint selected'
          : null
  const trySubmit = () => {
    if (problem) toast.info(problem)
    else if (!submit.isPending) submit.mutate()
  }

  const allJobs = useTrackedJobs()
  const jobs = allJobs.filter((j) => j.family === family && j.kind === 'inpaint')
  const useSettings = (job: TrackedJob) => {
    patchGenerationDraft('inpaint', family, paramsFromRequest(job.request))
    toast.success('Settings loaded into the form')
  }

  const showNegative = info.negativePrompt || params.trueCfgScale > 1
  const settings = (
    <GenerationSettings
      family={family}
      params={params}
      onChange={setParams}
      showSize={false}
      showBatch={false}
      extraSampling={
        family === 'flux' && (
          <SliderField
            label="Strength"
            value={ui.strength}
            onChange={(strength) => setUi({ strength })}
            min={0.05}
            max={1}
            step={0.05}
            hint="How much the masked area may change. 1 repaints it completely."
          />
        )
      }
    />
  )

  const submitButton = (
    <Button
      className="flex-1"
      size="lg"
      onClick={trySubmit}
      disabled={submit.isPending}
      aria-describedby="inpaint-submit-hint"
      title={problem ?? undefined}
    >
      {submit.isPending ? <Loader2 className="animate-spin" /> : <Brush />}
      Inpaint
    </Button>
  )

  return (
    <div className="grid flex-1 gap-6 pb-24 lg:grid-cols-[minmax(20rem,24rem)_1fr] lg:pb-0">
      <aside className="order-2 space-y-5 lg:order-none lg:sticky lg:top-20 lg:max-h-[calc(100svh-6.5rem)] lg:self-start lg:overflow-y-auto lg:pr-2">
        <PromptField
          label="Prompt"
          value={params.prompt}
          onChange={(prompt) => setParams({ prompt })}
          placeholder="What should appear in the masked area?"
          onSubmit={trySubmit}
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

        <div className="hidden lg:block">{settings}</div>

        <div className="sticky bottom-0 hidden space-y-1.5 bg-background pt-2 pb-1 lg:block">
          <div className="flex">{submitButton}</div>
          <p id="inpaint-submit-hint" className="text-center text-xs text-muted-foreground">
            {problem ?? (
              <>
                <Kbd>{MOD_KEY}</Kbd> + <Kbd>Enter</Kbd> in a prompt to submit
              </>
            )}
          </p>
        </div>
      </aside>

      <div className="order-1 min-w-0 space-y-8 lg:order-none">
        <section aria-label="Image and mask">
          {image ? (
            <div className="space-y-2">
              <MaskEditor
                key={image.url}
                ref={editorRef}
                image={image}
                initialMask={workspaces.get(family)?.mask}
                onMaskChange={setHasMask}
                onPersist={persistMask}
                onReplaceImage={() => setImage(undefined)}
              />
              <p className="text-xs text-muted-foreground">
                {image.name} · working at {image.width}×{image.height}
                {(image.originalWidth !== image.width || image.originalHeight !== image.height) &&
                  ` (scaled from ${image.originalWidth}×${image.originalHeight})`}
              </p>
            </div>
          ) : (
            <ImageDropzone maxSide={maxSideFor(family)} onImage={setImage} />
          )}
        </section>

        <section aria-label="Results">
          {jobs.length > 0 && <h2 className="mb-3 text-sm font-medium text-muted-foreground">Results</h2>}
          <ImageResults
            jobs={jobs}
            onUseSettings={useSettings}
            emptyState={null}
            renderDetails={(job, result) => {
              const source = jobSources.get(job.id)
              return source ? <BeforeAfter before={source} after={result.imageUrl} alt={job.label} /> : null
            }}
          />
        </section>
      </div>

      {/* Mobile: settings in a bottom sheet, submit always reachable. */}
      <div className="fixed inset-x-0 bottom-0 z-20 flex gap-2 border-t bg-background/90 p-3 backdrop-blur-md lg:hidden">
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="outline" size="lg" aria-label="Inpainting settings">
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
