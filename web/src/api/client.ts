import createClient from 'openapi-fetch'

import type { paths } from './schema'

/**
 * Typed client for the sd-api-backend.
 *
 * Requests go to the SPA's own origin (`/api/...`): the Vite dev server and
 * the production reverse proxy forward them to the backend, so no CORS setup
 * is needed and the backend URL is a deployment concern, not a build input.
 * The origin is spelled out because `Request` outside browsers (tests) rejects
 * relative URLs.
 */
export const api = createClient<paths>({ baseUrl: window.location.origin })

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(status: number, detail: unknown) {
    super(formatDetail(status, detail))
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/** FastAPI errors carry `detail` as a string or a list of validation errors. */
function formatDetail(status: number, detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) => [d.loc?.slice(1).join('.'), d.msg].filter(Boolean).join(': '))
      .join('; ')
  }
  return `Request failed with status ${status}`
}

/**
 * Unwrap an openapi-fetch result, throwing ApiError on non-2xx responses so
 * TanStack Query sees failures as errors.
 */
export async function unwrap<T>(request: Promise<{ data?: T; error?: unknown; response: Response }>): Promise<T> {
  const { data, error, response } = await request
  if (!response.ok) {
    const detail = (error as { detail?: unknown } | undefined)?.detail ?? error
    throw new ApiError(response.status, detail)
  }
  return data as T
}
