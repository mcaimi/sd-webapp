import { useState, type ReactNode } from 'react'

import { useModels } from '@/api/queries'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import type { FamilyId } from '@/lib/families'

import { ensureSafetensors } from './merge-utils'

/** Whether an output name would overwrite an existing checkpoint (the backend doesn't check). */
export function useWouldOverwrite(family: FamilyId, outputName: string): boolean {
  const { data = [] } = useModels(family, 'checkpoints')
  const name = ensureSafetensors(outputName)
  return !!name && data.some((m) => m.name === name)
}

/**
 * Wraps a submit: when the output would overwrite an existing checkpoint, ask
 * first. Returns the trigger and the dialog to render.
 */
export function useOverwriteConfirm(outputName: string, overwrite: boolean, onConfirm: () => void) {
  const [open, setOpen] = useState(false)
  const request = () => (overwrite ? setOpen(true) : onConfirm())
  const dialog: ReactNode = (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Overwrite {ensureSafetensors(outputName)}?</AlertDialogTitle>
          <AlertDialogDescription>
            A checkpoint with this name already exists. The merge will replace it, and this can't be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Keep existing</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={onConfirm}>
            Overwrite
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
  return { request, dialog }
}
