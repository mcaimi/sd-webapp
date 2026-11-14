import { NotFoundPage } from '@/routes/not-found-page'
import { useFamilyParam } from '@/routes/use-family-param'

import { ModelsPage } from './models-page'

export function Component() {
  const family = useFamilyParam()
  return family ? <ModelsPage key={family} family={family} /> : <NotFoundPage />
}
