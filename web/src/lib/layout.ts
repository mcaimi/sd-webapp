import { useCallback, type CSSProperties } from 'react'

import { useDraft } from './drafts'

/**
 * Desktop column widths, as percentages of the viewport: the navigation
 * sidebar, the generation parameters, and the results (the remainder).
 * Persisted so a resized layout survives reloads.
 */
export type Layout = { menu: number; params: number }

export const DEFAULT_LAYOUT: Layout = { menu: 20, params: 50 }
export const MENU_RANGE = [12, 35] as const
export const MIN_PARAMS = 20
export const MIN_RESULTS = 15

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

export const maxParams = (menu: number) => 100 - menu - MIN_RESULTS

/** Growing the menu takes room from the results first, then the parameters. */
export function withMenu(layout: Layout, menu: number): Layout {
  const next = clamp(menu, ...MENU_RANGE)
  return { menu: next, params: clamp(layout.params, MIN_PARAMS, maxParams(next)) }
}

export function withParams(layout: Layout, params: number): Layout {
  return { ...layout, params: clamp(params, MIN_PARAMS, maxParams(layout.menu)) }
}

export function useLayout() {
  const [stored, setStored] = useDraft<Layout>('layout', () => DEFAULT_LAYOUT)
  // Re-clamp on read in case the stored values came from an older range.
  const layout = withParams(withMenu(stored, stored.menu), stored.params)
  const setMenu = useCallback((menu: number) => setStored(withMenu(layout, menu)), [layout, setStored])
  const setParams = useCallback((params: number) => setStored(withParams(layout, params)), [layout, setStored])
  return { layout, setMenu, setParams }
}

/**
 * Grid template for tool pages: parameters, resize handle, results. The
 * parameters take their share of the space left beside the navigation.
 */
export const PARAMS_GRID = 'lg:grid-cols-[minmax(18rem,var(--params-width))_auto_minmax(0,1fr)]'

export function useParamsGridStyle(): CSSProperties {
  const { layout } = useLayout()
  return { '--params-width': `${(layout.params / (100 - layout.menu)) * 100}%` } as CSSProperties
}
