import type { LucideIcon } from 'lucide-react'

import { Card, CardContent } from '@/components/ui/card'

interface StatTileProps {
  label: string
  value: string
  detail?: string
  icon?: LucideIcon
}

export function StatTile({ label, value, detail, icon: Icon }: StatTileProps) {
  return (
    <Card className="gap-0 py-4">
      <CardContent className="space-y-1 px-4">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>{label}</span>
          {Icon && <Icon className="size-4" aria-hidden />}
        </div>
        <p className="truncate text-2xl font-semibold tracking-tight">{value}</p>
        {detail && <p className="truncate text-xs text-muted-foreground">{detail}</p>}
      </CardContent>
    </Card>
  )
}
