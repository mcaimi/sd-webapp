// Download the backend's OpenAPI document into openapi.json.
// The committed snapshot is what types are generated from, so builds and CI
// don't need a running backend. Re-run after backend API changes:
//   SD_API_URL=http://host:8000 npm run api:sync
import { writeFile } from 'node:fs/promises'

const base = (process.env.SD_API_URL || 'http://localhost:8000').replace(/\/$/, '')
const res = await fetch(`${base}/openapi.json`)
if (!res.ok) {
  console.error(`Failed to fetch ${base}/openapi.json: ${res.status} ${res.statusText}`)
  process.exit(1)
}
const spec = await res.json()
await writeFile(new URL('../openapi.json', import.meta.url), JSON.stringify(spec, null, 2) + '\n')
console.log(`openapi.json updated from ${base} (${spec.info?.title} ${spec.info?.version})`)
