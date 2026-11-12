import { NotFoundPage } from '@/routes/not-found-page'
import { useFamilyParam } from '@/routes/use-family-param'

import { GeneratePage } from './generate-page'

export function Component() {
  const family = useFamilyParam()
  // Keyed by family so switching families starts from that family's own draft.
  return family ? <GeneratePage key={family} family={family} /> : <NotFoundPage />
}
