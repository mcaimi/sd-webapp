import { workingSize } from './image-size'
import type { SourceImage } from './source-images'

/** Decode an image file and redraw it at its working size as PNG. */
export async function loadSourceImage(file: File, maxSide: number): Promise<SourceImage> {
  const bitmap = await createImageBitmap(file)
  const original = { width: bitmap.width, height: bitmap.height }
  const size = workingSize(original.width, original.height, maxSide)
  const canvas = document.createElement('canvas')
  canvas.width = size.width
  canvas.height = size.height
  const ctx = canvas.getContext('2d')!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(bitmap, 0, 0, size.width, size.height)
  bitmap.close()
  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode image'))), 'image/png'),
  )
  const dataUrl = await new Promise<string>((resolve) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.readAsDataURL(blob)
  })
  return {
    url: URL.createObjectURL(blob),
    base64: dataUrl.slice(dataUrl.indexOf(',') + 1),
    width: size.width,
    height: size.height,
    originalWidth: original.width,
    originalHeight: original.height,
    name: file.name || 'pasted image',
  }
}
