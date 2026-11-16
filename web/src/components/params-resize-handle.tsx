import { DEFAULT_LAYOUT, MIN_PARAMS, maxParams, useLayout } from '@/lib/layout'

import { ResizeHandle } from './resize-handle'

/** Sits between the parameters and results columns of a `PARAMS_GRID` layout. */
export function ParamsResizeHandle() {
  const { layout, setParams } = useLayout()
  const share = 100 - layout.menu

  return (
    <ResizeHandle
      label="Resize parameters"
      value={layout.params}
      min={MIN_PARAMS}
      max={maxParams(layout.menu)}
      onDrag={(x, handle) => {
        const grid = handle.parentElement!.getBoundingClientRect()
        setParams(((x - grid.left) / grid.width) * share)
      }}
      onStep={(delta) => setParams(layout.params + delta)}
      onReset={() => setParams(DEFAULT_LAYOUT.params)}
      className="hidden lg:sticky lg:top-20 lg:flex lg:h-[calc(100svh-6.5rem)]"
    />
  )
}
