import { useMutation } from '@tanstack/react-query'
import { GitMerge } from 'lucide-react'
import { useEffect } from 'react'
import { toast } from 'sonner'

import { useModels } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { submitMerge } from '@/jobs/submit'
import { useDraft } from '@/lib/drafts'
import type { FamilyId } from '@/lib/families'

import { AlphaField, MethodPicker, ModelField, OutputNameField, SubmitRow } from './merge-controls'
import { useOverwriteConfirm, useWouldOverwrite } from './merge-hooks'
import { ensureSafetensors, suggestOutputName, type MergeMethod } from './merge-utils'

interface SingleDraft {
  base: string
  target: string
  method: MergeMethod
  alpha: number
  /** Empty while the user hasn't typed a name: the suggestion follows the inputs. */
  customName: string
  preserveMetadata: boolean
}

export function SingleMergeForm({ family }: { family: FamilyId }) {
  const [draft, setDraft] = useDraft<SingleDraft>(`merge-single:${family}`, () => ({
    base: '',
    target: '',
    method: 'linear',
    alpha: 0.5,
    customName: '',
    preserveMetadata: true,
  }))
  const { data: checkpoints } = useModels(family, 'checkpoints')

  // Preselect the first two checkpoints, and drop selections that disappeared.
  useEffect(() => {
    if (!checkpoints) return
    const names = checkpoints.map((m) => m.name)
    const base = names.includes(draft.base) ? draft.base : (names[0] ?? '')
    const target =
      names.includes(draft.target) && draft.target !== base ? draft.target : (names.find((n) => n !== base) ?? '')
    if (base !== draft.base || target !== draft.target) setDraft({ base, target })
  }, [checkpoints, draft.base, draft.target, setDraft])

  const outputName = draft.customName || suggestOutputName(draft.base, draft.target, draft.method, draft.alpha)
  const overwrite = useWouldOverwrite(family, outputName)

  const submit = useMutation({
    mutationFn: () =>
      submitMerge(family, {
        base: draft.base,
        target: draft.target,
        method: draft.method,
        alpha: draft.alpha,
        outputName: ensureSafetensors(outputName),
        preserveMetadata: draft.preserveMetadata,
      }),
    onSuccess: () => toast.success('Merge queued', { description: ensureSafetensors(outputName) }),
    onError: (error) => toast.error('Could not start merge', { description: error.message }),
  })
  const confirm = useOverwriteConfirm(outputName, overwrite, () => submit.mutate())

  const problem = !checkpoints?.length
    ? 'No checkpoints installed for this family'
    : !draft.base || !draft.target
      ? 'Pick a base and a target model'
      : draft.base === draft.target
        ? 'Base and target must differ'
        : !ensureSafetensors(outputName)
          ? 'Enter an output filename'
          : null

  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2">
        <ModelField family={family} label="Base model (A)" value={draft.base} onChange={(base) => setDraft({ base })} />
        <ModelField
          family={family}
          label="Target model (B)"
          value={draft.target}
          onChange={(target) => setDraft({ target })}
          exclude={[draft.base]}
        />
      </div>
      <MethodPicker value={draft.method} onChange={(method) => setDraft({ method })} />
      <AlphaField method={draft.method} value={draft.alpha} onChange={(alpha) => setDraft({ alpha })} />
      <div className="space-y-1">
        <OutputNameField family={family} value={outputName} onChange={(customName) => setDraft({ customName })} />
        {draft.customName && (
          <Button variant="link" size="xs" className="px-0" onClick={() => setDraft({ customName: '' })}>
            Use suggested name
          </Button>
        )}
      </div>
      <div className="flex items-center gap-2">
        <Switch
          id="single-preserve"
          checked={draft.preserveMetadata}
          onCheckedChange={(preserveMetadata) => setDraft({ preserveMetadata })}
        />
        <Label htmlFor="single-preserve" className="font-normal">
          Keep the base model's metadata
        </Label>
      </div>
      <SubmitRow
        problem={problem}
        pending={submit.isPending}
        onSubmit={confirm.request}
        label="Merge"
        icon={<GitMerge />}
      />
      {confirm.dialog}
    </div>
  )
}
