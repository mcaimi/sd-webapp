import { NotFoundPage } from '@/routes/not-found-page'
import { useFamilyParam } from '@/routes/use-family-param'

import { MergePage } from './merge-page'

export function Component() {
  const family = useFamilyParam()
  return family ? <MergePage key={family} family={family} /> : <NotFoundPage />
}
