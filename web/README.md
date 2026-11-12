# SD Studio (web)

Single-page frontend for [sd-api-backend](https://github.com/mcaimi/sd-api-backend), replacing the Streamlit UI.

**Stack:** Vite · React 19 · TypeScript · Tailwind CSS 4 · shadcn/ui (Radix) · Motion · TanStack Query ·
React Router · openapi-fetch (types generated from the backend's OpenAPI document).

## Development

Start the backend, then:

```sh
npm install
npm run dev            # http://localhost:5173
```

The app calls the API on its own origin (`/api/...`). Vite proxies those requests to `SD_API_URL`
(default `http://localhost:8000`):

```sh
SD_API_URL=http://gpu-host:8000 npm run dev
```

No GPU or models? `npm run dev:mock` runs the same UI against `scripts/mock-api.mjs`, a dependency-free
stand-in for the backend: a sequential job queue with the real status semantics, generated PNGs, persisted gallery
outputs and merges. A prompt containing `fail` makes the job fail; `MOCK_SPEED=0.2` slows everything down.

In production, serve `dist/` behind a reverse proxy that forwards `/api` to the backend the same way;
no CORS configuration or build-time API URL is needed.

## Scripts

| Script                      | What it does                                                      |
| --------------------------- | ----------------------------------------------------------------- |
| `dev` / `build` / `preview` | Vite dev server, production build, preview the build              |
| `dev:mock`                  | Dev server backed by the mock API                                 |
| `test`                      | Vitest unit tests                                                 |
| `e2e`                       | Playwright end-to-end tests (production build + mock backend)     |
| `typecheck` / `lint`        | `tsc -b` / oxlint                                                 |
| `format` / `format:check`   | Prettier                                                          |
| `api:sync`                  | Download `openapi.json` from `SD_API_URL` and regenerate types    |
| `api:gen` / `api:check`     | Regenerate `src/api/schema.d.ts` / verify it matches the snapshot |

After a backend API change, run `npm run api:sync` and commit both `openapi.json` and
`src/api/schema.d.ts`. CI fails if they drift apart.

## Testing

- `npm test`: Vitest unit and hook tests (`src/**/*.test.ts`).
- `npm run e2e`: Playwright drives the production build (`vite preview`) against the mock backend, on desktop
  and a mobile viewport. Every page is scanned with axe in both themes and fails on serious WCAG 2.1 AA
  violations; any console error fails a test. The mock is reset before each test (`POST /__mock/reset`).
  - Use an installed Chromium instead of Playwright's download: `CHROMIUM_PATH=/path/to/chromium npm run e2e`.
  - Test a deployment instead of `vite preview`: `E2E_BASE_URL=http://localhost:8080 npm run e2e`. The deployment
    must proxy `/api` to the mock on port 8020. This also enables `e2e/container.spec.ts` (security headers,
    caching, SPA fallback, upload limits).

## Deployment

`Dockerfile` builds the app and serves it from unprivileged nginx (`nginx/default.conf.template`) on port 8080:

- `/api/` is proxied to `SD_API_URL` (substituted at container start), streamed so model downloads aren't buffered.
- Unknown paths fall back to `index.html`. Hashed `/assets/` are cached as immutable; everything else is
  `no-cache`.
- Uploads up to 64 MB are accepted, since inpainting posts base64 images.
- A strict CSP is set: no inline scripts, which is why the pre-paint theme script lives in `public/theme-init.js`.
- `/healthz` serves the probes.

`deploy/` holds Kustomize manifests (Deployment, Service, OpenShift Route with a 300 s timeout for downloads).

## Layout

```
src/
  api/          client.ts (typed fetch client), queries.ts (TanStack Query hooks),
                types.ts (hand-written shapes the OpenAPI spec leaves untyped), schema.d.ts (generated)
  jobs/         job-store.ts (jobs submitted from this browser, persisted), use-job-sync.ts (polling,
                toasts, cache invalidation), job-results.ts (result fetching), submit.ts (API submissions)
  features/     one folder per page (generate, inpaint, merge, gallery, system); each route.tsx is lazy-loaded
  lib/          families.ts (model family registry), generation.ts (form ⇄ API mapping), drafts.ts (persisted forms)
  components/   app-shell/, jobs/ (job cards, dock, results feed), form/, pickers/, lightbox, ui/ (shadcn, generated)
  routes/       router.tsx and shell-level pages
```

### Jobs

The backend runs everything as queued jobs but doesn't know who submitted what, so the client tracks its own
jobs in `jobs/job-store.ts` (localStorage, so the queue and recent results survive reloads). One hook in the app
shell polls `GET /jobs/`, which returns statuses without images, while anything is active. It then fetches each
completed job once and turns the image into an object URL.

The backend reports no progress until a job finishes, so running jobs show elapsed time and an indeterminate bar,
never a percentage. Cancelling a running job doesn't stop it on the backend, so the UI only offers Cancel for
queued jobs. Results expire on the backend after an hour; persisted images remain in the gallery.

**Model families** (`sd15`, `sdxl`, `illustrious`, `flux`) are described once in `src/lib/families.ts`
(defaults, resolution presets, scheduler overrides, Flux's `true_cfg_scale`). Pages are written once and
parameterised by family through the URL (`/generate/:family`). Adding a family the backend supports is a
registry entry, not a new page.

shadcn components are added with `npx shadcn@latest add <component>` and live in `src/components/ui`;
they are excluded from lint and formatting so regenerated files stay diffable.
