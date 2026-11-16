import { Globe, HardDrive } from 'lucide-react'

import { useCivitaiStatus } from '@/api/queries'
import { PageHeader } from '@/components/page-header'
import { QueryError } from '@/components/query-error'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useDraft } from '@/lib/drafts'
import { FAMILIES, RESOURCE_LABELS, type FamilyId, type ResourceType } from '@/lib/families'

import { CivitaiBrowser, type BrowseFilters } from './civitai-browser'
import { DownloadList } from './download-list'
import { EmptyState } from './empty-state'
import { LocalModels } from './local-models'

type Tab = 'local' | 'civitai'
const RESOURCES = Object.keys(RESOURCE_LABELS) as ResourceType[]

export function ModelsPage({ family }: { family: FamilyId }) {
  const [ui, setUi] = useDraft(`models-ui:${family}`, () => ({
    tab: 'local' as Tab,
    resource: 'checkpoints' as ResourceType,
    filters: {} as BrowseFilters,
  }))
  const status = useCivitaiStatus()
  const civitaiEnabled = !!status.data?.enabled

  return (
    <>
      <PageHeader
        title="Models"
        description={`${FAMILIES[family].label} checkpoints, LoRAs and VAEs installed on the backend, and more to download from Civitai.`}
      />
      <div className="space-y-4">
        <DownloadList family={family} />
        <Tabs value={ui.tab} onValueChange={(tab) => setUi({ tab: tab as Tab })}>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <TabsList>
              <TabsTrigger value="local">
                <HardDrive />
                Installed
              </TabsTrigger>
              <TabsTrigger value="civitai">
                <Globe />
                Browse Civitai
              </TabsTrigger>
            </TabsList>
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={ui.resource}
              onValueChange={(resource) => resource && setUi({ resource: resource as ResourceType })}
              aria-label="Model type"
            >
              {RESOURCES.map((resource) => (
                <ToggleGroupItem key={resource} value={resource} className="px-3">
                  {RESOURCE_LABELS[resource]}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>
          <TabsContent value="local" className="mt-4">
            <LocalModels
              family={family}
              resource={ui.resource}
              onBrowse={civitaiEnabled ? () => setUi({ tab: 'civitai' }) : undefined}
            />
          </TabsContent>
          <TabsContent value="civitai" className="mt-4">
            {status.error ? (
              <QueryError error={status.error} onRetry={() => status.refetch()} title="Could not reach Civitai" />
            ) : status.isPending ? (
              <Skeleton className="h-64 rounded-xl" />
            ) : !status.data?.enabled ? (
              <EmptyState
                icon={Globe}
                title="Civitai is turned off"
                description="The backend has the Civitai integration disabled. Enable civitai.enabled in its settings to browse and download models."
              />
            ) : (
              <CivitaiBrowser
                family={family}
                resource={ui.resource}
                status={status.data}
                filters={ui.filters}
                onFiltersChange={(patch) => setUi({ filters: { ...ui.filters, ...patch } })}
              />
            )}
          </TabsContent>
        </Tabs>
      </div>
    </>
  )
}
