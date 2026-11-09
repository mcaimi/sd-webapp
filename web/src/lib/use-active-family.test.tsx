import { act, render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { describe, expect, it } from 'vitest'

import { useActiveFamily } from './use-active-family'

function Probe() {
  const [family, setFamily] = useActiveFamily()
  return (
    <>
      <output>{family}</output>
      <button onClick={() => setFamily('illustrious')}>switch</button>
    </>
  )
}

function renderAt(path: string) {
  const router = createMemoryRouter(
    [
      { path: '/:tool/:family', element: <Probe /> },
      { path: '*', element: <Probe /> },
    ],
    { initialEntries: [path] },
  )
  render(<RouterProvider router={router} />)
  return router
}

describe('useActiveFamily', () => {
  it('takes the family from a tool URL and remembers it', () => {
    renderAt('/generate/flux')
    expect(screen.getByRole('status')).toHaveTextContent('flux')
    expect(localStorage.getItem('sd-webapp.family')).toBe('flux')
  })

  it('falls back to the remembered family off tool routes', () => {
    localStorage.setItem('sd-webapp.family', 'sd15')
    renderAt('/system')
    expect(screen.getByRole('status')).toHaveTextContent('sd15')
  })

  it('ignores unknown families in the URL', () => {
    renderAt('/generate/bogus')
    expect(screen.getByRole('status')).toHaveTextContent('sdxl')
  })

  it('navigates to the same tool when switching family', async () => {
    const router = renderAt('/inpaint/sdxl')
    await act(async () => screen.getByRole('button').click())
    expect(router.state.location.pathname).toBe('/inpaint/illustrious')
    expect(screen.getByRole('status')).toHaveTextContent('illustrious')
  })

  it('updates in place off tool routes', async () => {
    const router = renderAt('/gallery')
    await act(async () => screen.getByRole('button').click())
    expect(router.state.location.pathname).toBe('/gallery')
    expect(screen.getByRole('status')).toHaveTextContent('illustrious')
  })
})
