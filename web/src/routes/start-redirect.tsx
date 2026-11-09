import { Navigate } from 'react-router'

import { toolPath } from '@/lib/navigation'
import { readStoredFamily } from '@/lib/use-active-family'

/** `/` opens the generate tool for the last-used family. */
export function StartRedirect() {
  return <Navigate to={toolPath('generate', readStoredFamily())} replace />
}
