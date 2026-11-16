import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

export function NsfwBadge({ className }: { className?: string }) {
  return (
    <Badge
      variant="outline"
      className={cn('h-4 shrink-0 px-1 text-[10px] text-destructive', className)}
      title="Marked NSFW on Civitai"
    >
      NSFW
    </Badge>
  )
}
