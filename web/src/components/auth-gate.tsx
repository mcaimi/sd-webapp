import { KeyRound, LoaderCircle } from 'lucide-react'
import { useEffect, useState, type FormEvent, type ReactNode } from 'react'

import { auth, useAuthState } from '@/api/auth'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

/** Renders the app once the session is valid, and asks for the access token otherwise. */
export function AuthGate({ children }: { children: ReactNode }) {
  const state = useAuthState()

  useEffect(() => {
    void auth.check()
  }, [])

  if (state === 'checking') return null
  if (state === 'signed-out') return <TokenPrompt />
  return children
}

function TokenPrompt() {
  const [token, setToken] = useState('')
  const [error, setError] = useState<string>()
  const [pending, setPending] = useState(false)

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    setPending(true)
    setError(undefined)
    try {
      if (!(await auth.signIn(token))) setError("That token isn't valid.")
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setPending(false)
    }
  }

  return (
    <main className="flex min-h-svh items-center justify-center p-4">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>
            <h1 className="flex items-center gap-2 text-lg font-semibold">
              <KeyRound className="size-5 text-muted-foreground" aria-hidden />
              SD Studio
            </h1>
          </CardTitle>
          <CardDescription>
            Enter the access token. It's set with <code>SD_WEB_TOKEN</code>, or printed in the server log at startup
            when that isn't set.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-3">
            <div className="space-y-1.5">
              <Label htmlFor="access-token">Access token</Label>
              <Input
                id="access-token"
                type="password"
                autoComplete="current-password"
                autoFocus
                required
                value={token}
                onChange={(e) => setToken(e.target.value)}
                aria-invalid={error ? true : undefined}
                aria-describedby={error ? 'access-token-error' : undefined}
              />
              {error && (
                <p id="access-token-error" role="alert" className="text-sm text-destructive">
                  {error}
                </p>
              )}
            </div>
            <Button type="submit" className="w-full" disabled={pending || !token.trim()}>
              {pending && <LoaderCircle className="animate-spin" aria-hidden />}
              Continue
            </Button>
          </form>
        </CardContent>
      </Card>
    </main>
  )
}
