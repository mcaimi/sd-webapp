// `npm run dev:mock`: the Vite dev server backed by the mock API instead of a
// real sd-api-backend. Useful without a GPU or models.
import { createServer } from 'vite'

import { startMockApi } from './mock-api.mjs'

const mockPort = Number(process.env.MOCK_PORT ?? 8010)
await startMockApi(mockPort)
process.env.SD_API_URL = `http://localhost:${mockPort}`

const server = await createServer({ server: { port: Number(process.env.PORT ?? 5173) } })
await server.listen()
server.printUrls()
console.log(`  ➜  Mock API: http://localhost:${mockPort}`)
