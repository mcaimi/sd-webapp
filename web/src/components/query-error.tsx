import { RotateCw, TriangleAlert } from 'lucide-react'

import { Button } from '@/components/ui/button'

interface QueryErrorProps {
  error: Error
  onRetry?: () => void
  title?: string
}

export function QueryError({ error, onRetry, title = 'Could not load data' }: QueryErrorProps) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm"
    >
      <TriangleAlert className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden />
      <div className="min-w-0 flex-1 space-y-0.5">
        <p className="font-medium">{title}</p>
        <p className="break-words text-muted-foreground">{error.message}</p>
      </div>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RotateCw />
          Retry
        </Button>
      )}
    </div>
  )
}
