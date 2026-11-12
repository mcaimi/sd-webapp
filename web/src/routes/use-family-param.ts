import { useParams } from 'react-router'

import { isFamilyId, type FamilyId } from '@/lib/families'

/** The `:family` route param, or undefined when it isn't a supported family. */
export function useFamilyParam(): FamilyId | undefined {
  const { family } = useParams()
  return isFamilyId(family) ? family : undefined
}
