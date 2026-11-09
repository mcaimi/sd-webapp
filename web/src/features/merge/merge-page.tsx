import { Trash2 } from 'lucide-react'

import { PageHeader } from '@/components/page-header'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { isActive, jobStore, useTrackedJobs, type TrackedJob } from '@/jobs/job-store'
import { useDraft } from '@/lib/drafts'
import { FAMILIES, type FamilyId } from '@/lib/families'

import { BatchMergeForm } from './batch-merge-form'
import { MergeResults } from './merge-results'
import { RecipeMergeForm } from './recipe-merge-form'
import { SingleMergeForm } from './single-merge-form'

type Mode = 'single' | 'batch' | 'recipe'
const MERGE_KINDS = new Set<TrackedJob['kind']>(['merge', 'batch-merge', 'recipe-merge'])

export function MergePage({ family }: { family: FamilyId }) {
  const [ui, setUi] = useDraft(`merge-ui:${family}`, () => ({ mode: 'single' as Mode }))
  const jobs = useTrackedJobs().filter((j) => j.family === family && MERGE_KINDS.has(j.kind))
  const hasFinished = jobs.some((j) => !isActive(j))

  return (
    <>
      <PageHeader title="Merge" description={`Blend ${FAMILIES[family].label} checkpoints into new ones.`} />
      <div className="grid gap-6 lg:grid-cols-[minmax(22rem,30rem)_1fr]">
        <Card className="self-start">
          <CardContent>
            <Tabs value={ui.mode} onValueChange={(mode) => setUi({ mode: mode as Mode })}>
              <TabsList className="mb-5 w-full">
                <TabsTrigger value="single">Single</TabsTrigger>
                <TabsTrigger value="batch">Batch</TabsTrigger>
                <TabsTrigger value="recipe">Recipe</TabsTrigger>
              </TabsList>
              <TabsContent value="single">
                <SingleMergeForm family={family} />
              </TabsContent>
              <TabsContent value="batch">
                <BatchMergeForm family={family} />
              </TabsContent>
              <TabsContent value="recipe">
                <RecipeMergeForm family={family} />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
        <section aria-label="Merge jobs" className="min-w-0 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-muted-foreground">Merge jobs</h2>
            {hasFinished && (
              <Button
                variant="ghost"
                size="xs"
                onClick={() => jobStore.clearFinished((j) => j.family === family && MERGE_KINDS.has(j.kind))}
              >
                <Trash2 />
                Clear finished
              </Button>
            )}
          </div>
          <MergeResults jobs={jobs} family={family} />
        </section>
      </div>
    </>
  )
}
