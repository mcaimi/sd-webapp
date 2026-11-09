import { useMutation } from '@tanstack/react-query'
import { ArrowDown, ArrowUp, FlaskConical, GripVertical, Plus, X } from 'lucide-react'
import { AnimatePresence, Reorder, useDragControls } from 'motion/react'
import { useEffect } from 'react'
import { toast } from 'sonner'

import { useModels } from '@/api/queries'
import { ModelCombobox } from '@/components/pickers/model-combobox'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { submitRecipeMerge } from '@/jobs/submit'
import { useDraft } from '@/lib/drafts'
import type { FamilyId } from '@/lib/families'

import { ModelField, OutputNameField, SubmitRow } from './merge-controls'
import { useOverwriteConfirm, useWouldOverwrite } from './merge-hooks'
import { ensureSafetensors, isBlend, MERGE_METHODS, type MergeMethod } from './merge-utils'

interface RecipeStep {
  id: string
  target: string
  method: MergeMethod
  alpha: number
}

interface RecipeDraft {
  base: string
  steps: RecipeStep[]
  outputName: string
}

const MAX_STEPS = 10

export function RecipeMergeForm({ family }: { family: FamilyId }) {
  const [draft, setDraft] = useDraft<RecipeDraft>(`merge-recipe:${family}`, () => ({
    base: '',
    steps: [],
    outputName: 'recipe_merge.safetensors',
  }))
  const { data: checkpoints = [], isSuccess } = useModels(family, 'checkpoints')

  useEffect(() => {
    if (isSuccess && !checkpoints.some((m) => m.name === draft.base)) setDraft({ base: checkpoints[0]?.name ?? '' })
  }, [checkpoints, isSuccess, draft.base, setDraft])

  const overwrite = useWouldOverwrite(family, draft.outputName)
  const submit = useMutation({
    mutationFn: () =>
      submitRecipeMerge(
        family,
        draft.base,
        draft.steps.map(({ target, method, alpha }) => ({ target, method, alpha })),
        ensureSafetensors(draft.outputName),
      ),
    onSuccess: () => toast.success('Recipe queued', { description: ensureSafetensors(draft.outputName) }),
    onError: (error) => toast.error('Could not start recipe', { description: error.message }),
  })
  const confirm = useOverwriteConfirm(draft.outputName, overwrite, () => submit.mutate())

  const setSteps = (steps: RecipeStep[]) => setDraft({ steps })
  const updateStep = (id: string, patch: Partial<RecipeStep>) =>
    setSteps(draft.steps.map((s) => (s.id === id ? { ...s, ...patch } : s)))
  const move = (index: number, delta: number) => {
    const next = [...draft.steps]
    const [step] = next.splice(index, 1)
    next.splice(index + delta, 0, step)
    setSteps(next)
  }
  const addStep = () =>
    setSteps([
      ...draft.steps,
      {
        id: crypto.randomUUID(),
        // Prefer a model the recipe doesn't use yet.
        target:
          (
            checkpoints.find((m) => m.name !== draft.base && !draft.steps.some((s) => s.target === m.name)) ??
            checkpoints.find((m) => m.name !== draft.base)
          )?.name ?? '',
        method: 'linear',
        alpha: 0.5,
      },
    ])

  const problem = !checkpoints.length
    ? 'No checkpoints installed for this family'
    : !draft.base
      ? 'Pick a base model'
      : draft.steps.length === 0
        ? 'Add at least one step'
        : draft.steps.some((s) => !s.target)
          ? 'Every step needs a target model'
          : !ensureSafetensors(draft.outputName)
            ? 'Enter an output filename'
            : null

  return (
    <div className="space-y-5">
      <ModelField family={family} label="Start from" value={draft.base} onChange={(base) => setDraft({ base })} />

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>
            Steps{' '}
            <span className="font-normal text-muted-foreground">
              {draft.steps.length}/{MAX_STEPS}
            </span>
          </Label>
          <Button variant="ghost" size="xs" onClick={addStep} disabled={draft.steps.length >= MAX_STEPS}>
            <Plus />
            Add step
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Applied in order: each step merges the result so far with its target. Drag to reorder.
        </p>
        {draft.steps.length === 0 ? (
          <button
            type="button"
            onClick={addStep}
            className="w-full rounded-lg border border-dashed p-6 text-sm text-muted-foreground hover:bg-accent/50"
          >
            Add the first step
          </button>
        ) : (
          <Reorder.Group axis="y" values={draft.steps} onReorder={setSteps} className="space-y-2">
            <AnimatePresence initial={false}>
              {draft.steps.map((step, index) => (
                <StepCard
                  key={step.id}
                  family={family}
                  step={step}
                  index={index}
                  count={draft.steps.length}
                  exclude={[draft.base]}
                  onChange={(patch) => updateStep(step.id, patch)}
                  onMove={(delta) => move(index, delta)}
                  onRemove={() => setSteps(draft.steps.filter((s) => s.id !== step.id))}
                />
              ))}
            </AnimatePresence>
          </Reorder.Group>
        )}
      </div>

      <OutputNameField family={family} value={draft.outputName} onChange={(outputName) => setDraft({ outputName })} />
      <SubmitRow
        problem={problem}
        pending={submit.isPending}
        onSubmit={confirm.request}
        label={draft.steps.length > 1 ? `Run ${draft.steps.length}-step recipe` : 'Run recipe'}
        icon={<FlaskConical />}
      />
      {confirm.dialog}
    </div>
  )
}

function StepCard({
  family,
  step,
  index,
  count,
  exclude,
  onChange,
  onMove,
  onRemove,
}: {
  family: FamilyId
  step: RecipeStep
  index: number
  count: number
  exclude: string[]
  onChange: (patch: Partial<RecipeStep>) => void
  onMove: (delta: number) => void
  onRemove: () => void
}) {
  const drag = useDragControls()
  return (
    <Reorder.Item
      value={step}
      dragListener={false}
      dragControls={drag}
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0 }}
      className="relative list-none rounded-lg border bg-card p-3"
    >
      <div className="flex items-start gap-2">
        <button
          type="button"
          onPointerDown={(e) => drag.start(e)}
          className="mt-1.5 cursor-grab touch-none text-muted-foreground active:cursor-grabbing"
          aria-label="Drag to reorder"
          tabIndex={-1}
        >
          <GripVertical className="size-4" />
        </button>
        <div className="min-w-0 flex-1 space-y-3">
          <div className="flex items-center gap-2">
            <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium">
              {index + 1}
            </span>
            <div className="min-w-0 flex-1">
              <ModelCombobox
                family={family}
                resource="checkpoints"
                value={step.target}
                onChange={(target) => onChange({ target })}
                exclude={exclude}
                placeholder="Target model"
              />
            </div>
          </div>
          <div className="grid grid-cols-[8rem_1fr] items-center gap-3">
            <Select value={step.method} onValueChange={(method) => onChange({ method: method as MergeMethod })}>
              <SelectTrigger size="sm" className="w-full" aria-label={`Step ${index + 1} method`}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MERGE_METHODS.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="flex items-center gap-2">
              <Slider
                value={[step.alpha]}
                min={0}
                max={1}
                step={0.05}
                onValueChange={([alpha]) => onChange({ alpha })}
                aria-label={`Step ${index + 1} ${isBlend(step.method) ? 'mix' : 'strength'}`}
              />
              <span className="w-9 text-right text-xs text-muted-foreground tabular-nums">{step.alpha.toFixed(2)}</span>
            </div>
          </div>
        </div>
        <div className="flex flex-col">
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => onMove(-1)}
            disabled={index === 0}
            aria-label={`Move step ${index + 1} up`}
          >
            <ArrowUp />
          </Button>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => onMove(1)}
            disabled={index === count - 1}
            aria-label={`Move step ${index + 1} down`}
          >
            <ArrowDown />
          </Button>
          <Button variant="ghost" size="icon-xs" onClick={onRemove} aria-label={`Remove step ${index + 1}`}>
            <X />
          </Button>
        </div>
      </div>
    </Reorder.Item>
  )
}
