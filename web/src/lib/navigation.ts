import { Activity, Brush, GitMerge, Images, Sparkles, type LucideIcon } from 'lucide-react'

import type { FamilyId } from './families'

/** Tools that operate on one model family; their routes are `/<id>/:family`. */
export type ToolId = 'generate' | 'inpaint' | 'merge'

export interface ToolNavItem {
  id: ToolId
  label: string
  description: string
  icon: LucideIcon
}

export interface PageNavItem {
  path: string
  label: string
  description: string
  icon: LucideIcon
}

export const TOOLS: ToolNavItem[] = [
  { id: 'generate', label: 'Generate', description: 'Text to image', icon: Sparkles },
  { id: 'inpaint', label: 'Inpaint', description: 'Repaint masked regions', icon: Brush },
  { id: 'merge', label: 'Merge', description: 'Blend checkpoints', icon: GitMerge },
]

export const PAGES: PageNavItem[] = [
  { path: '/gallery', label: 'Gallery', description: 'Saved generations', icon: Images },
  { path: '/system', label: 'System', description: 'Backend health and resources', icon: Activity },
]

export function toolPath(tool: ToolId, family: FamilyId): string {
  return `/${tool}/${family}`
}

export function isToolId(value: string | undefined): value is ToolId {
  return TOOLS.some((t) => t.id === value)
}
