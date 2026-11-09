// Apply the stored theme before first paint (see src/components/theme-provider.tsx).
// A separate file rather than an inline script, so the CSP needs no 'unsafe-inline' for scripts.
try {
  const t = localStorage.getItem('theme') || 'dark'
  const dark = t === 'dark' || (t === 'system' && matchMedia('(prefers-color-scheme: dark)').matches)
  document.documentElement.classList.toggle('dark', dark)
  document.documentElement.style.colorScheme = dark ? 'dark' : 'light'
} catch {
  // Storage unavailable: keep the default theme.
}
