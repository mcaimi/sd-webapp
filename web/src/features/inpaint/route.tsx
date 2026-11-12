import { NotFoundPage } from '@/routes/not-found-page'
import { useFamilyParam } from '@/routes/use-family-param'

import { InpaintPage } from './inpaint-page'

export function Component() {
  const family = useFamilyParam()
  // Keyed by family so each family keeps its own draft and image.
  return family ? <InpaintPage key={family} family={family} /> : <NotFoundPage />
}
