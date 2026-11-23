import { useSyncExternalStore } from 'react'

import { queryClient } from './query-client'

export type AuthState = 'checking' | 'signed-in' | 'signed-out'

let state: AuthState = 'checking'
let pendingCheck: Promise<void> | undefined
const listeners = new Set<() => void>()

function setState(next: AuthState) {
  if (state === next) return
  state = next
  listeners.forEach((fn) => fn())
}

function sessionRequest(method: 'GET' | 'POST' | 'DELETE', token?: string) {
  return fetch(new URL('/auth/session', window.location.origin), {
    method,
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    cache: 'no-store',
  })
}

/**
 * Access token session. The server (nginx in production, the Vite dev and
 * preview servers otherwise) guards `/api` and trades the token for an
 * HttpOnly cookie, so the token is never kept in browser storage and plain
 * links (downloads) are authorized too.
 */
export const auth = {
  get: () => state,
  subscribe(fn: () => void) {
    listeners.add(fn)
    return () => {
      listeners.delete(fn)
    }
  },

  /**
   * Ask the server whether this browser's session is valid. It answers 200
   * either way, so a signed-out visit doesn't log a failed request. A server
   * without the gate (404) or an unreachable one leaves the app usable, and
   * its API errors show where they happen.
   */
  check(): Promise<void> {
    pendingCheck ??= sessionRequest('GET')
      .then(async (res) => {
        const session = res.ok ? ((await res.json()) as { authenticated?: boolean }) : undefined
        setState(session?.authenticated === false ? 'signed-out' : 'signed-in')
      })
      .catch(() => setState('signed-in'))
      .finally(() => {
        pendingCheck = undefined
      })
    return pendingCheck
  },

  /** Resolves to false when the server rejects the token. */
  async signIn(token: string): Promise<boolean> {
    const res = await sessionRequest('POST', token.trim())
    if (res.status === 401) return false
    if (!res.ok) throw new Error(`Sign-in failed with status ${res.status}`)
    setState('signed-in')
    // Anything that failed while signed out is fetched again.
    void queryClient.invalidateQueries()
    return true
  },

  async signOut() {
    const res = await sessionRequest('DELETE')
    if (res.ok) setState('signed-out')
  },
}

export function useAuthState() {
  return useSyncExternalStore(auth.subscribe, auth.get)
}
