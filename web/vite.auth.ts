import { randomBytes, timingSafeEqual } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'

import type { Connect, Logger, Plugin } from 'vite'

const COOKIE = 'sd_webapp_token'
const MAX_AGE = 30 * 24 * 60 * 60

/**
 * Access token gate for the dev and preview servers, mirroring the production
 * nginx config: `/api` needs the session cookie or an `Authorization: Bearer`
 * header, and `/auth/session` reports (GET), opens (POST with the header) or
 * closes (DELETE) the session. The token comes from SD_WEB_TOKEN, or is
 * generated and printed when the server starts.
 */
export function authGate(configured: string | undefined): Plugin {
  let token: string | undefined

  function resolveToken(logger: Logger): string {
    if (token) return token
    if (configured) {
      token = configured
    } else {
      token = randomBytes(32).toString('hex')
      logger.info(`\n  SD_WEB_TOKEN is not set; generated an access token:\n\n    ${token}\n`)
    }
    return token
  }

  return {
    name: 'sd-webapp-auth-gate',
    configureServer(server) {
      server.middlewares.use(middleware(resolveToken(server.config.logger)))
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware(resolveToken(server.config.logger)))
    },
  }
}

function middleware(token: string): Connect.NextHandleFunction {
  const matches = (value: string | undefined) => {
    if (value === undefined) return false
    const a = Buffer.from(value)
    const b = Buffer.from(token)
    return a.length === b.length && timingSafeEqual(a, b)
  }
  const bearer = (req: IncomingMessage) => matches(req.headers.authorization?.match(/^Bearer (.*)$/)?.[1])
  const authorized = (req: IncomingMessage) => bearer(req) || matches(readCookie(req, COOKIE))

  return (req, res, next) => {
    const path = req.url?.split('?')[0] ?? ''
    if (path === '/auth/session') return session(req, res)
    if (path.startsWith('/api') && !authorized(req)) return send(res, 401)
    next()
  }

  function session(req: IncomingMessage, res: ServerResponse) {
    res.setHeader('Cache-Control', 'no-store')
    switch (req.method) {
      case 'GET':
        res.setHeader('Content-Type', 'application/json')
        res.statusCode = 200
        return res.end(JSON.stringify({ authenticated: authorized(req) }))
      case 'POST':
        if (!bearer(req)) return send(res, 401)
        res.setHeader('Set-Cookie', `${COOKIE}=${token}; Path=/; Max-Age=${MAX_AGE}; HttpOnly; SameSite=Strict`)
        return send(res, 204)
      case 'DELETE':
        res.setHeader('Set-Cookie', `${COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict`)
        return send(res, 204)
      default:
        return send(res, 405)
    }
  }
}

function readCookie(req: IncomingMessage, name: string): string | undefined {
  for (const part of req.headers.cookie?.split(';') ?? []) {
    const [key, ...value] = part.trim().split('=')
    if (key === name) return value.join('=')
  }
  return undefined
}

function send(res: ServerResponse, status: number) {
  res.statusCode = status
  res.end()
}
