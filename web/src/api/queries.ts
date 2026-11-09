import { useQuery } from '@tanstack/react-query'

import { FAMILIES, type FamilyId, type ResourceType } from '@/lib/families'

import { api, unwrap } from './client'
import type { ModelEntry } from './types'

export const queryKeys = {
  health: ['health'] as const,
  systemInfo: ['system-info'] as const,
  schedulers: ['schedulers'] as const,
  models: (family: FamilyId, resource: ResourceType) => ['models', family, resource] as const,
  modelMetadata: (family: FamilyId, resource: ResourceType, name: string) =>
    ['model-metadata', family, resource, name] as const,
  jobs: ['jobs'] as const,
}

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => unwrap(api.GET('/api/v1/system/health')),
    refetchInterval: 10_000,
    retry: false,
  })
}

export function useSystemInfo(options: { refetchInterval?: number } = {}) {
  return useQuery({
    queryKey: queryKeys.systemInfo,
    queryFn: () => unwrap(api.GET('/api/v1/system/info')),
    refetchInterval: options.refetchInterval ?? 10_000,
  })
}

/** Schedulers valid for a family: the registry override, else the API list. */
export function useSchedulers(family: FamilyId) {
  const override = FAMILIES[family].schedulers
  const query = useQuery({
    queryKey: queryKeys.schedulers,
    queryFn: () => unwrap(api.GET('/api/v1/info/schedulers')),
    staleTime: 5 * 60_000,
    enabled: !override,
  })
  return override ? { ...query, data: override, isPending: false, isError: false } : query
}

export function useModels(family: FamilyId, resource: ResourceType) {
  return useQuery({
    queryKey: queryKeys.models(family, resource),
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/models/{model_type}/{resource_type}', {
          params: { path: { model_type: family, resource_type: resource } },
        }),
      ),
    select: (data) => data.models as unknown as ModelEntry[],
    staleTime: 60_000,
  })
}

export function useModelMetadata(family: FamilyId, resource: ResourceType, name: string | undefined) {
  return useQuery({
    queryKey: queryKeys.modelMetadata(family, resource, name ?? ''),
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/models/{model_type}/{resource_type}/{model_name}/metadata', {
          params: { path: { model_type: family, resource_type: resource, model_name: name! } },
        }),
      ),
    enabled: !!name,
    staleTime: 10 * 60_000,
  })
}

/** All backend jobs (without results, so cheap to poll). */
export function useJobs(options: { enabled?: boolean; refetchInterval?: number | false } = {}) {
  return useQuery({
    queryKey: queryKeys.jobs,
    queryFn: () => unwrap(api.GET('/api/v1/jobs/')),
    enabled: options.enabled ?? true,
    refetchInterval: options.refetchInterval ?? 2_000,
  })
}
