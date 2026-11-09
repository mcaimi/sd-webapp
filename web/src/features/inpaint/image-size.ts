import type { FamilyId } from '@/lib/families'

/** Longest side inpainting works at: near each architecture's native resolution. */
export function maxSideFor(family: FamilyId): number {
  return family === 'sd15' ? 768 : 1024
}

/**
 * Size to work at for an uploaded image: scaled down (never up) so the longest
 * side fits `maxSide`, aspect ratio kept, both sides rounded to multiples of 8
 * as the VAE requires.
 */
export function workingSize(width: number, height: number, maxSide: number): { width: number; height: number } {
  const scale = Math.min(1, maxSide / Math.max(width, height))
  const round8 = (v: number) => Math.max(8, Math.round((v * scale) / 8) * 8)
  return {
    width: Math.min(round8(width), Math.floor(maxSide / 8) * 8),
    height: Math.min(round8(height), Math.floor(maxSide / 8) * 8),
  }
}
