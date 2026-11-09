import type { FamilyId } from '@/lib/families'

/** An uploaded image, already scaled to its working size. */
export interface SourceImage {
  /** Object URL for display. */
  url: string
  /** PNG bytes as base64 (no data: prefix), as the API expects. */
  base64: string
  width: number
  height: number
  originalWidth: number
  originalHeight: number
  name: string
}

/**
 * Session-only state. Source images are too big to persist with tracked jobs,
 * so the before/after view is available for jobs submitted in this session.
 */
export const jobSources = new Map<string, string>()

/** The image (and painted mask) being edited per family, kept across navigation. */
export const workspaces = new Map<FamilyId, { image: SourceImage; mask?: ImageData }>()
