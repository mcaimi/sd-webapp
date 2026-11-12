/** Overlay colour painted onto the mask canvas (shown semi-transparent over the image). */
export const MASK_RGB: readonly [number, number, number] = [225, 29, 29]

/** Whether any pixel of an RGBA buffer has been painted (non-zero alpha). */
export function hasPaint(rgba: Uint8ClampedArray): boolean {
  for (let i = 3; i < rgba.length; i += 4) if (rgba[i] !== 0) return true
  return false
}

/** Invert painted/unpainted areas in place, keeping soft brush edges soft. */
export function invertMask(rgba: Uint8ClampedArray, rgb: readonly [number, number, number] = MASK_RGB): void {
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = rgb[0]
    rgba[i + 1] = rgb[1]
    rgba[i + 2] = rgb[2]
    rgba[i + 3] = 255 - rgba[i + 3]
  }
}

/** Push onto a bounded undo stack, dropping the oldest entry. */
export function pushBounded<T>(stack: T[], item: T, limit: number): void {
  stack.push(item)
  if (stack.length > limit) stack.shift()
}
