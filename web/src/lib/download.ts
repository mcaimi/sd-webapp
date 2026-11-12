/** Trigger a browser download for a URL (object URL or same-origin API path). */
export function download(url: string, filename: string) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.append(a)
  a.click()
  a.remove()
}
