import { keepPreviousData, useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api, unwrap } from '@/api/client'
import type { FamilyId, ResourceType } from '@/lib/families'

const PAGE_SIZE = 24

export interface CivitaiSearchParams {
  family: FamilyId
  resource: ResourceType
  query: string
  sort: string
  period: string
  nsfw: boolean
}

// Keys start with 'civitai-model' so finished downloads refresh which files are local.
export const civitaiKeys = {
  search: (params: CivitaiSearchParams) => ['civitai-search', params] as const,
  model: (id: number, family: FamilyId, nsfw: boolean) => ['civitai-model', id, family, nsfw] as const,
  layout: (versionId: number, fileId: number) => ['civitai-layout', versionId, fileId] as const,
}

/** Civitai search, paged by cursor. Previous results stay visible while filters change. */
export function useCivitaiSearch(params: CivitaiSearchParams, enabled: boolean) {
  return useInfiniteQuery({
    queryKey: civitaiKeys.search(params),
    queryFn: ({ pageParam }) =>
      unwrap(
        api.GET('/api/v1/civitai/search', {
          params: {
            query: {
              model_type: params.family,
              resource_type: params.resource,
              query: params.query.trim() || undefined,
              sort: params.sort,
              period: params.period,
              nsfw: params.nsfw,
              limit: PAGE_SIZE,
              cursor: pageParam,
            },
          },
        }),
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
    enabled,
  })
}

/** A Civitai model with its versions and files, checked against `family` and the local model directories. */
export function useCivitaiModel(id: number | undefined, family: FamilyId, nsfw: boolean) {
  return useQuery({
    queryKey: civitaiKeys.model(id ?? 0, family, nsfw),
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/civitai/models/{model_id}', {
          params: { path: { model_id: id! }, query: { model_type: family, nsfw } },
        }),
      ),
    enabled: id !== undefined,
    staleTime: 5 * 60_000,
  })
}

/** Components a Flux checkpoint on Civitai bundles. Read remotely on request, so only fetched once asked for. */
export function useFileLayout(versionId: number, fileId: number, requested: boolean) {
  return useQuery({
    queryKey: civitaiKeys.layout(versionId, fileId),
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/civitai/files/layout', { params: { query: { version_id: versionId, file_id: fileId } } }),
      ),
    enabled: requested,
    // A file's content never changes.
    staleTime: Infinity,
    retry: false,
  })
}

/** Look up a local model on Civitai by hash; a match writes its Civitai details next to the file. */
export function useIdentifyModel(family: FamilyId, resource: ResourceType) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      unwrap(
        api.POST('/api/v1/civitai/identify', {
          body: { model_type: family, resource_type: resource, model_name: name },
        }),
      ),
    onSuccess: (data) => {
      if (data.matched) queryClient.invalidateQueries({ queryKey: ['models', family, resource] })
    },
  })
}
