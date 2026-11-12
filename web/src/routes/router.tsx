import { createBrowserRouter } from 'react-router'

import { AppShell } from '@/components/app-shell/app-shell'

import { NotFoundPage } from './not-found-page'
import { StartRedirect } from './start-redirect'

// Feature pages are lazy-loaded so the first paint only ships the shell.
export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <StartRedirect /> },
      { path: 'generate/:family', lazy: () => import('@/features/generate/route') },
      { path: 'inpaint/:family', lazy: () => import('@/features/inpaint/route') },
      { path: 'merge/:family', lazy: () => import('@/features/merge/route') },
      { path: 'gallery', lazy: () => import('@/features/gallery/route') },
      { path: 'system', lazy: () => import('@/features/system/route') },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
])
