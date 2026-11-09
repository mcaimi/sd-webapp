const isMac = /Mac|iPhone|iPad/.test(navigator.userAgent)

/** Label for the primary modifier key: ⌘ on Apple platforms, Ctrl elsewhere. */
export const MOD_KEY = isMac ? '⌘' : 'Ctrl'
