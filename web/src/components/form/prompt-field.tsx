import { useId } from 'react'

import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { CLIP_TOKEN_LIMIT, estimateTokens } from '@/lib/generation'
import { cn } from '@/lib/utils'

interface PromptFieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  rows?: number
  /** Ctrl/Cmd+Enter */
  onSubmit?: () => void
  autoFocus?: boolean
}

export function PromptField({ label, value, onChange, placeholder, rows = 4, onSubmit, autoFocus }: PromptFieldProps) {
  const id = useId()
  const tokens = estimateTokens(value)
  const over = tokens > CLIP_TOKEN_LIMIT

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label htmlFor={id}>{label}</Label>
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className={cn('text-xs tabular-nums', over ? 'font-medium text-meter-warn' : 'text-muted-foreground')}
              tabIndex={0}
            >
              ~{tokens} / {CLIP_TOKEN_LIMIT}
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-60">
            Approximate CLIP token count. Text past the {CLIP_TOKEN_LIMIT}-token window may be ignored by SD models.
          </TooltipContent>
        </Tooltip>
      </div>
      <Textarea
        id={id}
        value={value}
        rows={rows}
        placeholder={placeholder}
        autoFocus={autoFocus}
        className="field-sizing-content max-h-64 min-h-20 resize-none"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && onSubmit) {
            e.preventDefault()
            onSubmit()
          }
        }}
      />
    </div>
  )
}
