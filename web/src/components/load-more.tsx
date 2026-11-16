import { Loader2 } from 'lucide-react'
import { useEffect, useRef } from 'react'

import { Button } from '@/components/ui/button'

/** "Load more" button for paged lists that also loads the next page as it scrolls into view. */
export function LoadMore({
  hasNextPage,
  isFetching,
  onLoadMore,
}: {
  hasNextPage: boolean
  isFetching: boolean
  onLoadMore: () => void
}) {
  const sentinel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = sentinel.current
    if (!el || !hasNextPage) return
    const observer = new IntersectionObserver(([entry]) => entry.isIntersecting && !isFetching && onLoadMore(), {
      rootMargin: '400px',
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [hasNextPage, isFetching, onLoadMore])

  if (!hasNextPage) return null
  return (
    <div ref={sentinel} className="flex justify-center py-6">
      <Button variant="outline" onClick={onLoadMore} disabled={isFetching}>
        {isFetching && <Loader2 className="animate-spin" />}
        {isFetching ? 'Loading…' : 'Load more'}
      </Button>
    </div>
  )
}
