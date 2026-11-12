import { Dices, Lock, LockOpen } from 'lucide-react'
import { motion, useAnimationControls } from 'motion/react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { randomSeed, SEED_MAX } from '@/lib/generation'

interface SeedFieldProps {
  seed: number
  locked: boolean
  onChange: (patch: { seed?: number; seedLocked?: boolean }) => void
}

/**
 * Unlocked: every image gets a fresh random seed. Locked: the seed is reused,
 * so changes to other settings can be compared on the same noise.
 */
export function SeedField({ seed, locked, onChange }: SeedFieldProps) {
  const dice = useAnimationControls()

  return (
    <div className="space-y-2">
      <Label htmlFor="seed">Seed</Label>
      <div className="flex gap-1.5">
        <Input
          id="seed"
          key={`${seed}-${locked}`}
          inputMode="numeric"
          defaultValue={locked ? seed : ''}
          placeholder="Random"
          disabled={!locked}
          className="h-8 tabular-nums"
          onBlur={(e) => {
            const value = Number(e.target.value)
            if (Number.isInteger(value) && value >= 0 && value <= SEED_MAX) onChange({ seed: value })
          }}
          onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        />
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="outline"
              size="icon"
              aria-label="Roll a new seed"
              onClick={() => {
                dice.start({ rotate: [0, 360], transition: { duration: 0.45, ease: 'easeOut' } })
                onChange({ seed: randomSeed(), seedLocked: true })
              }}
            >
              <motion.span animate={dice} className="flex">
                <Dices />
              </motion.span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>Roll and lock a new seed</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant={locked ? 'secondary' : 'outline'}
              size="icon"
              aria-pressed={locked}
              aria-label={locked ? 'Unlock seed' : 'Lock seed'}
              onClick={() => onChange({ seedLocked: !locked })}
            >
              {locked ? <Lock /> : <LockOpen />}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            {locked ? 'Seed locked: reused for every run' : 'Random seed for every image'}
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  )
}
