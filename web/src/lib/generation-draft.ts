import type { FamilyId } from './families'
import { useDraft, writeDraft } from './drafts'
import { defaultParams, type GenerationParams } from './generation'

type DraftTool = 'generate' | 'inpaint'

const key = (tool: DraftTool, family: FamilyId) => `${tool}:${family}`

export function useGenerationDraft(tool: DraftTool, family: FamilyId) {
  return useDraft<GenerationParams>(key(tool, family), () => defaultParams(family))
}

/** Write into a tool's form from elsewhere (remix, "use in generate"). */
export function patchGenerationDraft(tool: DraftTool, family: FamilyId, patch: Partial<GenerationParams>) {
  writeDraft<GenerationParams>(key(tool, family), () => defaultParams(family), patch)
}
