import { useCallback, useEffect, useSyncExternalStore } from 'react'
import { useMatch, useNavigate } from 'react-router'

import { DEFAULT_FAMILY, isFamilyId, type FamilyId } from './families'
import { isToolId, toolPath } from './navigation'

const STORAGE_KEY = 'sd-webapp.family'

// Shared store so every component using the hook sees the same family.
const listeners = new Set<() => void>()

export function readStoredFamily(): FamilyId {
  const stored = localStorage.getItem(STORAGE_KEY) ?? undefined
  return isFamilyId(stored) ? stored : DEFAULT_FAMILY
}

function writeStoredFamily(family: FamilyId) {
  if (localStorage.getItem(STORAGE_KEY) === family) return
  localStorage.setItem(STORAGE_KEY, family)
  listeners.forEach((notify) => notify())
}

function subscribe(notify: () => void) {
  listeners.add(notify)
  return () => listeners.delete(notify)
}

/**
 * The family the user is working with.
 *
 * On a tool route (`/generate/sdxl`) the URL is the source of truth, so links
 * are shareable; elsewhere the last-used family is remembered in localStorage.
 * Switching family while on a tool route navigates to the same tool.
 */
export function useActiveFamily(): [FamilyId, (family: FamilyId) => void] {
  const match = useMatch('/:tool/:family')
  const navigate = useNavigate()
  const tool = match?.params.tool
  const urlFamily = isToolId(tool) && isFamilyId(match?.params.family) ? match.params.family : undefined

  const storedFamily = useSyncExternalStore(subscribe, readStoredFamily)

  useEffect(() => {
    if (urlFamily) writeStoredFamily(urlFamily)
  }, [urlFamily])

  const setFamily = useCallback(
    (family: FamilyId) => {
      writeStoredFamily(family)
      if (urlFamily && isToolId(tool)) navigate(toolPath(tool, family))
    },
    [navigate, tool, urlFamily],
  )

  return [urlFamily ?? storedFamily, setFamily]
}
