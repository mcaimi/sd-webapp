import { useMutation } from '@tanstack/react-query'
import { Layers } from 'lucide-react'
import { useEffect } from 'react'
import { toast } from 'sonner'

import { useModels } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { submitBatchMerge } from '@/jobs/submit'
import { useDraft } from '@/lib/drafts'
import type { FamilyId } from '@/lib/families'
import { modelDisplayName } from '@/lib/models'

import { AlphaField, MethodPicker, ModelField, SubmitRow } from './merge-controls'
import type { MergeMethod } from './merge-utils'

interface BatchDraft {
  base: string
  targets: string[]
  method: MergeMethod
  alpha: number
  outputSubdir: string
  preserveMetadata: boolean
}

export function BatchMergeForm({ family }: { family: FamilyId }) {
  const [draft, setDraft] = useDraft<BatchDraft>(`merge-batch:${family}`, () => ({
    base: '',
    targets: [],
    method: 'linear',
    alpha: 0.5,
    outputSubdir: 'batch_merged',
    preserveMetadata: true,
  }))
  const { data: checkpoints = [], isSuccess } = useModels(family, 'checkpoints')
  const candidates = checkpoints.filter((m) => m.name !== draft.base)

  useEffect(() => {
    if (!isSuccess) return
    const names = checkpoints.map((m) => m.name)
    const base = names.includes(draft.base) ? draft.base : (names[0] ?? '')
    const targets = draft.targets.filter((t) => names.includes(t) && t !== base)
    if (base !== draft.base || targets.length !== draft.targets.length) setDraft({ base, targets })
  }, [checkpoints, isSuccess, draft.base, draft.targets, setDraft])

  const submit = useMutation({
    mutationFn: () =>
      submitBatchMerge(family, {
        base: draft.base,
        targets: draft.targets,
        method: draft.method,
        alpha: draft.alpha,
        outputSubdir: draft.outputSubdir.trim() || 'batch_merged',
        preserveMetadata: draft.preserveMetadata,
      }),
    onSuccess: () => toast.success('Batch merge queued', { description: `${draft.targets.length} outputs` }),
    onError: (error) => toast.error('Could not start batch merge', { description: error.message }),
  })

  const problem = !checkpoints.length
    ? 'No checkpoints installed for this family'
    : !draft.base
      ? 'Pick a base model'
      : draft.targets.length === 0
        ? 'Pick at least one target model'
        : null
  const toggle = (name: string, on: boolean) =>
    setDraft({ targets: on ? [...draft.targets, name] : draft.targets.filter((t) => t !== name) })
  const allSelected = candidates.length > 0 && draft.targets.length === candidates.length

  return (
    <div className="space-y-5">
      <ModelField family={family} label="Base model (A)" value={draft.base} onChange={(base) => setDraft({ base })} />

      <fieldset className="space-y-2">
        <div className="flex items-center justify-between">
          <legend className="text-sm font-medium">
            Target models (B){' '}
            <span className="font-normal text-muted-foreground">
              {draft.targets.length}/{candidates.length}
            </span>
          </legend>
          {candidates.length > 1 && (
            <Button
              variant="ghost"
              size="xs"
              onClick={() => setDraft({ targets: allSelected ? [] : candidates.map((m) => m.name) })}
            >
              {allSelected ? 'Select none' : 'Select all'}
            </Button>
          )}
        </div>
        <div className="max-h-56 space-y-1 overflow-y-auto rounded-lg border p-2">
          {candidates.length === 0 && (
            <p className="p-1 text-sm text-muted-foreground">No other checkpoints to merge with.</p>
          )}
          {candidates.map((m) => (
            <Label
              key={m.name}
              className="flex cursor-pointer items-center gap-2 rounded-md p-1.5 font-normal hover:bg-accent"
            >
              <Checkbox
                checked={draft.targets.includes(m.name)}
                onCheckedChange={(on) => toggle(m.name, on === true)}
              />
              <span className="truncate">{modelDisplayName(m.name)}</span>
            </Label>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          Each target is merged with the base separately
          {draft.targets.length > 0 &&
            `: ${draft.targets.length} new checkpoint${draft.targets.length === 1 ? '' : 's'}`}
          .
        </p>
      </fieldset>

      <MethodPicker value={draft.method} onChange={(method) => setDraft({ method })} />
      <AlphaField method={draft.method} value={draft.alpha} onChange={(alpha) => setDraft({ alpha })} />

      <div className="space-y-2">
        <Label htmlFor="batch-subdir">Output folder</Label>
        <Input
          id="batch-subdir"
          value={draft.outputSubdir}
          onChange={(e) => setDraft({ outputSubdir: e.target.value })}
          placeholder="batch_merged"
        />
        <p className="text-xs text-muted-foreground">Created inside the family's checkpoint folder.</p>
      </div>
      <div className="flex items-center gap-2">
        <Switch
          id="batch-preserve"
          checked={draft.preserveMetadata}
          onCheckedChange={(preserveMetadata) => setDraft({ preserveMetadata })}
        />
        <Label htmlFor="batch-preserve" className="font-normal">
          Keep the base model's metadata
        </Label>
      </div>
      <SubmitRow
        problem={problem}
        pending={submit.isPending}
        onSubmit={() => submit.mutate()}
        label={draft.targets.length > 1 ? `Create ${draft.targets.length} merges` : 'Start batch merge'}
        icon={<Layers />}
      />
    </div>
  )
}
