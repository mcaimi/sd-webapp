// Mock sd-api-backend for UI development and end-to-end tests.
//
// Mirrors the real backend's behaviour where the UI depends on it: a single
// sequential GPU-lane worker, jobs going queued -> running -> completed/failed/
// cancelled, progress that jumps from 0 to 1 on completion, cancelled running
// jobs finishing with their result discarded, the same result shapes, model
// lists with Civitai details and previews, and persisted outputs with the
// backend's metadata layout. Images are generated gradients encoded as real PNGs.
//
// Knobs:  MOCK_PORT (8010), MOCK_SPEED (1 = realistic-ish, 0.2 = fast).
// A prompt containing "fail" makes the job fail; "slow" makes it 5x slower.
// Civitai: a generated catalogue per family and resource type. Downloads run
// on a separate "io" lane with progress, taking ~2 s per GB at any speed; a file named "*-broken*" fails, a
// Flux file named "*-locked*" needs an API key to check its components.
import { randomUUID } from 'node:crypto'
import http from 'node:http'
import { deflateSync } from 'node:zlib'

const SPEED = Number(process.env.MOCK_SPEED ?? 1)
const FAMILIES = ['sd15', 'sdxl', 'illustrious', 'flux']
const MS_PER_STEP = { sd15: 60, sdxl: 110, illustrious: 110, flux: 180 }
const SD_SCHEDULERS = ['DPM++ 2M', 'DPM++ SDE', 'DPM2', 'Euler a', 'Euler', 'Heun', 'LMS']

const models = {
  sd15: {
    checkpoints: ['dreamshaper_8.safetensors', 'realisticVision_v60.safetensors', 'deliberate_v3.safetensors'],
    loras: ['add_detail.safetensors', 'film_grain.safetensors', 'watercolor_style.safetensors'],
    vaes: ['vae-ft-mse-840000.safetensors'],
  },
  sdxl: {
    checkpoints: ['sd_xl_base_1.0.safetensors', 'juggernautXL_v9.safetensors', 'dreamshaperXL_turbo.safetensors'],
    loras: ['xl_more_art.safetensors', 'xl_cinematic.safetensors'],
    vaes: ['sdxl_vae.safetensors'],
  },
  illustrious: {
    checkpoints: ['illustriousXL_v01.safetensors', 'noobai_xl_v1.safetensors'],
    loras: ['flat_color.safetensors', 'lineart_boost.safetensors'],
    vaes: [],
  },
  flux: {
    checkpoints: ['flux1-dev.safetensors', 'flux1-schnell.safetensors'],
    loras: ['flux_realism.safetensors'],
    vaes: ['ae.safetensors'],
  },
}

// Civitai sidecar details for some models, as the backend lists them.
const civitai = {
  'dreamshaper_8.safetensors': {
    model_id: 4384,
    version_id: 128713,
    model_name: 'DreamShaper',
    version_name: '8',
    base_model: 'SD 1.5',
    trained_words: [],
    nsfw: false,
    url: 'https://civitai.com/models/4384?modelVersionId=128713',
  },
  'watercolor_style.safetensors': {
    model_id: 1001,
    version_id: 2002,
    model_name: 'Watercolor Style',
    version_name: 'v2',
    base_model: 'SD 1.5',
    trained_words: ['wcstyle', 'watercolor painting'],
    nsfw: false,
    url: 'https://civitai.com/models/1001?modelVersionId=2002',
  },
  'realisticVision_v60.safetensors': {
    model_id: 4201,
    version_id: 245598,
    model_name: 'Realistic Vision',
    version_name: 'V6.0 B1',
    base_model: 'SD 1.5',
    trained_words: [],
    nsfw: true,
    url: 'https://civitai.com/models/4201?modelVersionId=245598',
  },
}
// Flux checkpoints: which components each file bundles.
const fluxLayouts = {
  'flux1-dev.safetensors': {
    transformer: true,
    clip_l: false,
    t5xxl: false,
    vae: false,
    format: 'diffusers',
    variant: 'dev',
  },
  'flux1-schnell.safetensors': {
    transformer: true,
    clip_l: true,
    t5xxl: true,
    vae: true,
    format: 'comfy',
    variant: 'schnell',
  },
}

function modelEntry(family, resource, name) {
  return {
    name,
    path: `models/${resource}/${family}/${name}`,
    civitai: civitai[name] ?? null,
    has_preview: name in civitai,
    flux_layout: family === 'flux' && resource === 'checkpoints' ? (fluxLayouts[name] ?? null) : null,
  }
}

// Snapshot so tests can reset merges back to the original model lists.
const initialModels = structuredClone(models)
const initialCivitai = structuredClone(civitai)

// ---------------------------------------------------------------- PNG encoding
const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})
function crc32(buf) {
  let c = 0xffffffff
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}
function chunk(type, data) {
  const len = Buffer.alloc(4)
  len.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(body))
  return Buffer.concat([len, body, crc])
}
function encodePng(width, height, pixel) {
  const raw = Buffer.alloc((width * 3 + 1) * height)
  let i = 0
  for (let y = 0; y < height; y++) {
    raw[i++] = 0
    for (let x = 0; x < width; x++) {
      const [r, g, b] = pixel(x, y)
      raw[i++] = r
      raw[i++] = g
      raw[i++] = b
    }
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr.set([8, 2, 0, 0, 0], 8)
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 6 })),
    chunk('IEND', Buffer.alloc(0)),
  ])
}
function hash(str) {
  let h = 2166136261
  for (const ch of str) h = Math.imul(h ^ ch.charCodeAt(0), 16777619)
  return h >>> 0
}
function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l)
  const f = (n) => {
    const k = (n + h / 30) % 12
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))))
  }
  return [f(0), f(8), f(4)]
}
function renderImage(width, height, seed, prompt) {
  const h = hash(`${seed}:${prompt}`)
  const c1 = hsl(h % 360, 0.65, 0.55)
  const c2 = hsl((h >> 9) % 360, 0.6, 0.25)
  const cx = 0.25 + ((h >> 3) % 50) / 100
  const cy = 0.25 + ((h >> 5) % 50) / 100
  const freq = 4 + ((h >> 7) % 8)
  const png = encodePng(width, height, (x, y) => {
    const u = x / width
    const v = y / height
    const d = Math.min(1, Math.hypot(u - cx, v - cy) * 1.6)
    const wave = 0.08 * Math.sin((u + v) * freq * Math.PI)
    const t = Math.min(1, Math.max(0, d + wave))
    return [0, 1, 2].map((k) => Math.round(c1[k] * (1 - t) + c2[k] * t))
  })
  return png.toString('base64')
}
function pngSize(base64) {
  const buf = Buffer.from(base64.slice(0, 64), 'base64')
  return buf.length >= 24 ? { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) } : { width: 512, height: 512 }
}

// ---------------------------------------------------------------- job queue
const jobs = new Map()
const outputs = [] // newest first: { id, created_at, png: Buffer, metadata }
let running = null

function submit(jobType, durationMs, run) {
  const job = {
    job_id: randomUUID(),
    job_type: jobType,
    lane: 'gpu',
    cancellable: false,
    status: 'queued',
    progress: 0,
    detail: null,
    created_at: new Date().toISOString(),
    started_at: null,
    completed_at: null,
    result: null,
    error: null,
    durationMs: durationMs / SPEED,
    run,
  }
  jobs.set(job.job_id, job)
  queueMicrotask(pump)
  return job
}
function pump() {
  if (running) return
  const next = [...jobs.values()].find((j) => j.status === 'queued' && j.lane === 'gpu')
  if (!next) return
  running = next
  next.status = 'running'
  next.started_at = new Date().toISOString()
  setTimeout(() => {
    // Like the real GPU worker: cancelling a running job doesn't stop it, but
    // its result is discarded and it stays cancelled.
    try {
      const result = next.run()
      if (next.status !== 'cancelled') {
        next.result = result
        next.status = 'completed'
        next.progress = 1
      }
    } catch (e) {
      if (next.status !== 'cancelled') {
        next.status = 'failed'
        next.error = e.message
      }
    }
    next.completed_at = new Date().toISOString()
    running = null
    pump()
  }, next.durationMs)
}
// eslint-disable-next-line no-unused-vars -- strip internal fields
const publicJob = ({ durationMs, run, download, ...job }) => job

// ---------------------------------------------------------------- handlers
function generationJob(family, req, kind) {
  const loras = (req.loras ?? []).map((l, i) => [
    `lora_${i}`,
    { name: l.name, lora_path: `models/loras/${family}/${l.name}`, merge_strength: l.strength },
  ])
  const steps = req.steps ?? 30
  const slow = req.positive_prompt?.includes('slow') ? 5 : 1
  return submit(kind === 'inpaint' ? 'inpaint' : 'txt2img', steps * MS_PER_STEP[family] * slow + 400, () => {
    if (req.positive_prompt?.includes('fail')) throw new Error('CUDA out of memory (mock failure)')
    const size = kind === 'inpaint' ? pngSize(req.image_base64) : { width: req.width, height: req.height }
    const seed = req.seed >= 0 ? req.seed : Math.floor(Math.random() * 2 ** 32)
    const image = renderImage(size.width, size.height, seed, req.positive_prompt)
    const metadata = {
      instances: [
        {
          prompt: req.positive_prompt,
          negative_prompt: req.negative_prompt ?? '',
          num_inference_steps: steps,
          width: size.width,
          height: size.height,
          guidance_scale: req.cfg_scale,
          seed,
          scheduler: req.scheduler,
        },
      ],
    }
    outputs.unshift({
      id: randomUUID(),
      created_at: new Date().toISOString(),
      png: Buffer.from(image, 'base64'),
      metadata: { generation: metadata, loras: Object.fromEntries(loras), vae: req.custom_vae ?? null },
    })
    return { image_base64: image, metadata, seed }
  })
}

function requireModel(res, family, name) {
  if (!models[family].checkpoints.includes(name)) {
    send(res, 404, { detail: `Model not found: ${name}` })
    return false
  }
  return true
}

function validationError(res, field) {
  send(res, 422, { detail: [{ type: 'missing', loc: ['body', field], msg: 'Field required', input: null }] })
}

const routes = []
const route = (method, pattern, handler) =>
  routes.push({ method, re: new RegExp(`^${pattern.replace(/:(\w+)/g, '(?<$1>[^/]+)')}/?$`), handler })

route('GET', '/api/v1/system/health', (_req, res) => send(res, 200, { status: 'healthy', service: 'sd-api-backend' }))
let cpu = 12
route('GET', '/api/v1/system/info', (_req, res) => {
  cpu = Math.min(100, Math.max(2, cpu + (running ? 18 : -6) + (Math.random() - 0.5) * 10))
  const all = [...jobs.values()]
  const gpuUsed = running ? 9.5 * 2 ** 30 + Math.random() * 2 ** 30 : 2.1 * 2 ** 30
  send(res, 200, {
    device: 'cuda',
    dtype: 'torch.float16',
    vendor: 'nvidia',
    gpu_name: 'NVIDIA RTX 4090 (mock)',
    cache_stats: { cached_models: 1, max_models: 2, in_use: running ? 1 : 0, cache_keys: ['sdxl:juggernautXL_v9'] },
    queue_stats: {
      queue_size: all.filter((j) => j.status === 'queued').length,
      total_jobs: all.length,
      num_workers: 1,
      workers_running: true,
      lanes: {
        gpu: {
          queue_size: all.filter((j) => j.status === 'queued').length,
          num_workers: 1,
          running_jobs: running ? 1 : 0,
        },
        io: { queue_size: 0, num_workers: 2, running_jobs: 0 },
      },
    },
    gpu_stats: {
      available: true,
      used_bytes: gpuUsed,
      total_bytes: 24 * 2 ** 30,
      total_is_approximate: false,
      compute: { available: true, utilization_percent: running ? 85 + Math.random() * 14 : Math.random() * 4 },
    },
    host_stats: {
      cpu_percent: cpu,
      cpu_count: 16,
      memory_used_bytes: 21 * 2 ** 30 + Math.random() * 2 ** 29,
      memory_total_bytes: 64 * 2 ** 30,
      memory_percent: 33 + Math.random(),
      load_average: [1.2, 1.4, 1.3],
    },
  })
})
route('GET', '/api/v1/civitai/status', (_req, res) =>
  send(res, 200, {
    enabled: true,
    has_api_key: false,
    nsfw_default: false,
    model_types: {
      sd15: ['SD 1.4', 'SD 1.5'],
      sdxl: ['SDXL 1.0', 'Pony'],
      illustrious: ['Illustrious', 'NoobAI'],
      flux: ['Flux.1 D', 'Flux.1 S'],
    },
    resource_types: { checkpoints: ['Checkpoint'], loras: ['LORA', 'LoCon', 'DoRA'], vaes: ['VAE'] },
    sort_options: ['Highest Rated', 'Most Downloaded', 'Newest'],
    period_options: ['AllTime', 'Year', 'Month', 'Week', 'Day'],
  }),
)
route('GET', '/api/v1/info/schedulers', (_req, res) => send(res, 200, SD_SCHEDULERS))

route('GET', '/api/v1/models/:family/:resource', (_req, res, { family, resource }) => {
  if (!models[family]?.[resource]) return send(res, 400, { detail: 'Invalid model type or resource' })
  send(res, 200, {
    model_type: family,
    resource_type: resource,
    models: models[family][resource].map((name) => modelEntry(family, resource, name)),
  })
})
route('GET', '/api/v1/models/:family/:resource/:name/preview', (_req, res, { family, resource, name }) => {
  name = decodeURIComponent(name)
  if (!models[family]?.[resource]?.includes(name)) return send(res, 404, { detail: 'Model not found' })
  if (!(name in civitai)) return send(res, 404, { detail: 'No preview for this model' })
  res.writeHead(200, { 'content-type': 'image/png' })
  res.end(Buffer.from(renderImage(128, 128, civitai[name].version_id, name), 'base64'))
})
route('GET', '/api/v1/models/:family/:resource/:name/metadata', (_req, res, { family, resource, name }) => {
  name = decodeURIComponent(name)
  if (!models[family]?.[resource]?.includes(name)) return send(res, 404, { detail: 'Model not found' })
  send(res, 200, {
    name,
    path: `models/${resource}/${family}/${name}`,
    metadata: {
      'modelspec.title': name.replace('.safetensors', ''),
      'modelspec.architecture': family === 'sd15' ? 'stable-diffusion-v1' : `${family}-base`,
      ss_base_model_version: family,
      ss_training_comment: 'Mock metadata from scripts/mock-api.mjs',
    },
  })
})
route('GET', '/api/v1/models/:family/:resource/:name/download', (_req, res, { family, resource, name }) => {
  name = decodeURIComponent(name)
  if (!models[family]?.[resource]?.includes(name)) return send(res, 404, { detail: 'Model not found' })
  res.writeHead(200, {
    'content-type': 'application/octet-stream',
    'content-disposition': `attachment; filename="${name}"`,
  })
  res.end(Buffer.alloc(1024))
})

for (const family of FAMILIES) {
  route('POST', `/api/v1/${family}/generate`, (req, res) => {
    const body = req.body
    if (!body.positive_prompt) return validationError(res, 'positive_prompt')
    if (!body.model_checkpoint) return validationError(res, 'model_checkpoint')
    if (!requireModel(res, family, body.model_checkpoint)) return
    const job = generationJob(family, body, 'generate')
    send(res, 200, { job_id: job.job_id, status: 'queued', message: 'Generation job submitted successfully' })
  })
  route('POST', `/api/v1/${family}/inpaint`, (req, res) => {
    const body = req.body
    for (const field of ['positive_prompt', 'model_checkpoint', 'image_base64', 'mask_base64'])
      if (!body[field]) return validationError(res, field)
    if (!requireModel(res, family, body.model_checkpoint)) return
    const job = generationJob(family, body, 'inpaint')
    send(res, 200, { job_id: job.job_id, status: 'queued', message: 'Inpainting job submitted successfully' })
  })
  route('POST', `/api/v1/${family}/generate/batch`, (req, res) => {
    const ids = req.body.requests.map((r) => generationJob(family, r, 'generate').job_id)
    send(res, 200, { job_ids: ids, message: `Submitted ${ids.length} generation jobs` })
  })
  route('POST', `/api/v1/${family}/generate/compare`, (req, res) => {
    const { models: names, ...rest } = req.body
    const comparisons = names.map((model) => ({
      model,
      job_id: generationJob(family, { ...rest, model_checkpoint: model }, 'generate').job_id,
    }))
    send(res, 200, { comparisons, message: `Submitted ${comparisons.length} comparison jobs` })
  })
}

route('GET', '/api/v1/jobs', (_req, res) =>
  send(
    res,
    200,
    // The real endpoint omits results so it stays cheap to poll.
    // eslint-disable-next-line no-unused-vars
    [...jobs.values()].map(({ result, ...j }) => publicJob(j)),
  ),
)
route('GET', '/api/v1/jobs/:id', (_req, res, { id }) => {
  const job = jobs.get(id)
  if (job) send(res, 200, publicJob(job))
  else send(res, 404, { detail: 'Job not found' })
})
route('DELETE', '/api/v1/jobs/:id', (_req, res, { id }) => {
  const job = jobs.get(id)
  if (!job || ['completed', 'failed', 'cancelled'].includes(job.status))
    return send(res, 400, { detail: 'Job cannot be cancelled (not found or already finished)' })
  // A queued job is finished now; a running one when its worker gets to the end.
  if (job.status === 'queued') job.completed_at = new Date().toISOString()
  job.status = 'cancelled'
  send(res, 200, { message: 'Job cancelled successfully', job_id: id })
})

function mergeOutput(family, name) {
  const file = name.endsWith('.safetensors') ? name : `${name}.safetensors`
  if (!models[family].checkpoints.includes(file)) models[family].checkpoints.push(file)
  return file
}
route('POST', '/api/v1/merge/merge', (req, res) => {
  const b = req.body
  if (!requireModel(res, b.model_type, b.base_model) || !requireModel(res, b.model_type, b.target_model)) return
  const job = submit('merge', 4000, () => {
    const file = mergeOutput(b.model_type, b.output_name)
    return {
      output_path: `models/checkpoints/${b.model_type}/${file}`,
      base_model: b.base_model,
      target_model: b.target_model,
      method: b.method,
      alpha: b.alpha,
      model_type: b.model_type,
    }
  })
  send(res, 200, { job_id: job.job_id, status: 'queued', message: `Merge job submitted: ${b.output_name}` })
})
route('POST', '/api/v1/merge/batch', (req, res) => {
  const b = req.body
  if (!requireModel(res, b.model_type, b.base_model)) return
  const job = submit('batch_merge', 2500 * b.target_models.length, () => {
    const paths = b.target_models.map((t) => {
      const file = mergeOutput(b.model_type, `merged_${b.method}_${b.base_model.replace('.safetensors', '')}_${t}`)
      return `models/checkpoints/${b.model_type}/${b.output_subdir}/${file}`
    })
    return { output_paths: paths, ...b, count: paths.length }
  })
  send(res, 200, { job_id: job.job_id, status: 'queued', message: 'Batch merge job submitted' })
})
route('POST', '/api/v1/merge/recipe', (req, res) => {
  const b = req.body
  if (!requireModel(res, b.model_type, b.base_model)) return
  const job = submit('recipe_merge', 2500 * b.steps.length, () => {
    const file = mergeOutput(b.model_type, b.output_name)
    return {
      output_path: `models/checkpoints/${b.model_type}/${file}`,
      base_model: b.base_model,
      steps: b.steps,
      model_type: b.model_type,
      step_count: b.steps.length,
    }
  })
  send(res, 200, { job_id: job.job_id, status: 'queued', message: 'Recipe merge job submitted' })
})

route('GET', '/api/v1/outputs', (req, res) => {
  const url = new URL(req.url, 'http://x')
  const limit = Number(url.searchParams.get('limit') ?? 24)
  const offset = Number(url.searchParams.get('offset') ?? 0)
  send(res, 200, {
    total: outputs.length,
    limit,
    offset,
    items: outputs.slice(offset, offset + limit).map((o) => ({
      id: o.id,
      created_at: o.created_at,
      size_bytes: o.png.length,
      image_url: `/api/v1/outputs/${o.id}/image`,
      json_url: `/api/v1/outputs/${o.id}/metadata`,
    })),
  })
})
route('GET', '/api/v1/outputs/:id/image', (_req, res, { id }) => {
  const o = outputs.find((x) => x.id === id)
  if (!o) return send(res, 404, { detail: 'Output image not found' })
  res.writeHead(200, { 'content-type': 'image/png' })
  res.end(o.png)
})
route('GET', '/api/v1/outputs/:id/metadata', (_req, res, { id }) => {
  const o = outputs.find((x) => x.id === id)
  if (o) send(res, 200, o.metadata)
  else send(res, 404, { detail: 'Output metadata not found' })
})
route('DELETE', '/api/v1/outputs/:id', (_req, res, { id }) => {
  const i = outputs.findIndex((x) => x.id === id)
  if (i < 0) return send(res, 404, { detail: 'Output not found' })
  outputs.splice(i, 1)
  res.writeHead(204)
  res.end()
})

// ---------------------------------------------------------------- civitai
const CIVITAI_BASE_MODELS = {
  sd15: ['SD 1.5'],
  sdxl: ['SDXL 1.0', 'Pony'],
  illustrious: ['Illustrious'],
  flux: ['Flux.1 D', 'Flux.1 S'],
}
const CIVITAI_TYPES = { checkpoints: 'Checkpoint', loras: 'LORA', vaes: 'VAE' }
const CATALOGUE_SIZE = 30
const SORTS = ['Highest Rated', 'Most Downloaded', 'Newest']
const PERIODS = ['AllTime', 'Year', 'Month', 'Week', 'Day']
const NAMES = ['Dream', 'Real', 'Anime', 'Painterly', 'Cinematic', 'Pixel', 'Ink', 'Neon', 'Velvet', 'Aurora']

const otherFamily = (family) => (family === 'sd15' ? 'sdxl' : 'sd15')
const mockImageUrl = (id) => `/api/v1/__mock/civitai-image/original=true/${id}.png`

/** Deterministic catalogue: ids encode family, resource and index so detail lookups need no state. */
function catalogue(family, resource) {
  const f = FAMILIES.indexOf(family)
  const r = Object.keys(CIVITAI_TYPES).indexOf(resource)
  return Array.from({ length: CATALOGUE_SIZE }, (_, i) => {
    const id = 100000 + f * 10000 + r * 1000 + i
    const nsfw = i % 7 === 3
    return {
      id,
      name: `${NAMES[i % NAMES.length]} ${resource === 'loras' ? 'Style' : family.toUpperCase()} ${Math.floor(i / NAMES.length) + 1}`,
      type: CIVITAI_TYPES[resource],
      creator: `artist${i % 5}`,
      nsfw,
      tags: ['style', resource === 'loras' ? 'concept' : 'base model'],
      stats: {
        downloads: (CATALOGUE_SIZE - i) * 1371,
        rating: 4 + (i % 10) / 10,
        rating_count: 40 + i,
        thumbs_up: (CATALOGUE_SIZE - i) * 97,
        favorites: i * 3,
      },
      preview: { url: mockImageUrl(id), type: 'image', nsfw_level: nsfw ? 8 : 1, width: 512, height: 682 },
      base_models: [CIVITAI_BASE_MODELS[family][0]],
      // For sorting by "Newest".
      published_at: new Date(Date.UTC(2026, 0, 1 + i)).toISOString(),
    }
  })
}

function findCatalogueModel(id) {
  for (const family of FAMILIES)
    for (const resource of Object.keys(CIVITAI_TYPES)) {
      const model = catalogue(family, resource).find((m) => m.id === id)
      if (model) return { model, family, resource }
    }
  return undefined
}

const fileName = (model, version, suffix = '') =>
  `${model.name.replace(/\W+/g, '')}_${version}${suffix}.safetensors`.replace(/_+/g, '_')

/** Versions of a catalogue model, as GET /civitai/models/{id} normalizes them for `browsing` family. */
function versionsFor(model, family, resource, browsing) {
  const versionId = (n) => model.id * 10 + n
  const fileId = (n, k) => versionId(n) * 10 + k
  const words = resource === 'loras' ? [`${model.name.split(' ')[0].toLowerCase()}style`] : []
  const versions = [
    { n: 1, name: 'v2.0', family, base: CIVITAI_BASE_MODELS[family][0] },
    { n: 2, name: 'v1.0', family, base: CIVITAI_BASE_MODELS[family][0] },
    {
      n: 3,
      name: `v1.0 ${otherFamily(family).toUpperCase()}`,
      family: otherFamily(family),
      base: CIVITAI_BASE_MODELS[otherFamily(family)][0],
    },
  ]
  return versions.map((v) => {
    const warnings = v.family !== browsing ? [`Made for ${v.base} (${v.family}), not ${browsing}`] : []
    const files = [
      {
        id: fileId(v.n, 1),
        name: fileName(model, v.name.replace(/\W+/g, '')),
        size_bytes: resource === 'checkpoints' ? 2_132_625_408 : 151_000_000,
        type: 'Model',
        format: 'SafeTensor',
        fp: 'fp16',
        size_variant: 'pruned',
        sha256: `${model.id}${v.n}`.padEnd(64, '0'),
        primary: true,
        target_resource_type: resource,
        compatible: true,
        reason: null,
        warnings,
      },
      {
        id: fileId(v.n, 2),
        name: fileName(model, v.name.replace(/\W+/g, ''), '_full').replace('.safetensors', '.ckpt'),
        size_bytes: 4_265_250_816,
        type: 'Model',
        format: 'PickleTensor',
        fp: 'fp32',
        size_variant: 'full',
        sha256: `${model.id}${v.n}2`.padEnd(64, '0'),
        primary: false,
        target_resource_type: resource,
        compatible: false,
        reason: 'PickleTensor format not supported',
        warnings: [],
      },
    ]
    if (resource === 'checkpoints' && v.n === 1) {
      files.push({
        id: fileId(v.n, 3),
        name: `${model.name.replace(/\W+/g, '')}_vae.safetensors`,
        size_bytes: 334_643_268,
        type: 'VAE',
        format: 'SafeTensor',
        fp: 'fp16',
        size_variant: null,
        sha256: `${model.id}${v.n}3`.padEnd(64, '0'),
        primary: false,
        target_resource_type: 'vaes',
        compatible: true,
        reason: null,
        warnings,
      })
    }
    return {
      id: versionId(v.n),
      name: v.name,
      base_model: v.base,
      base_model_type: 'Standard',
      model_type: v.family,
      trained_words: words,
      published_at: new Date(Date.UTC(2026, 5, 10 - v.n)).toISOString(),
      early_access: false,
      preview: model.preview,
      files: files.map((file) => {
        const local = models[browsing]?.[file.target_resource_type]?.includes(file.name)
        return {
          ...file,
          local_path: local ? `models/${file.target_resource_type}/${browsing}/civitai/${file.name}` : null,
          local_match: local ? 'sha256' : null,
        }
      }),
    }
  })
}

function findFile(versionId) {
  const modelId = Math.floor(versionId / 10)
  const found = findCatalogueModel(modelId)
  if (!found) return undefined
  return { ...found, versions: versionsFor(found.model, found.family, found.resource, found.family) }
}

const query = (req) => new URL(req.url, 'http://x').searchParams

route('GET', '/api/v1/civitai/search', (req, res) => {
  const q = query(req)
  const family = q.get('model_type')
  const resource = q.get('resource_type')
  if (!FAMILIES.includes(family) || !CIVITAI_TYPES[resource])
    return send(res, 400, { detail: 'Invalid model_type or resource_type' })
  const sort = q.get('sort') ?? SORTS[0]
  if (!SORTS.includes(sort)) return send(res, 400, { detail: `Invalid sort. Must be one of ${JSON.stringify(SORTS)}` })
  if (!PERIODS.includes(q.get('period') ?? PERIODS[0]))
    return send(res, 400, { detail: `Invalid period. Must be one of ${JSON.stringify(PERIODS)}` })
  const text = (q.get('query') ?? '').trim().toLowerCase()
  const nsfw = q.get('nsfw') === 'true'
  const limit = Number(q.get('limit') ?? 24)
  const offset = Number(q.get('cursor') ?? 0)
  const items = catalogue(family, resource)
    .filter((m) => (nsfw || !m.nsfw) && (!text || m.name.toLowerCase().includes(text)))
    .sort((a, b) =>
      sort === 'Newest'
        ? b.published_at.localeCompare(a.published_at)
        : sort === 'Most Downloaded'
          ? b.stats.downloads - a.stats.downloads
          : b.stats.rating - a.stats.rating,
    )
  const page = items.slice(offset, offset + limit)
  send(res, 200, {
    // eslint-disable-next-line no-unused-vars -- strip internal fields
    items: page.map(({ published_at, ...m }) => m),
    next_cursor: offset + limit < items.length ? String(offset + limit) : null,
  })
})

route('GET', '/api/v1/civitai/models/:id', (req, res, { id }) => {
  const browsing = query(req).get('model_type')
  if (!FAMILIES.includes(browsing)) return send(res, 400, { detail: 'Invalid model_type' })
  const found = findCatalogueModel(Number(id))
  if (!found) return send(res, 404, { detail: 'Model not found on Civitai' })
  // eslint-disable-next-line no-unused-vars -- strip internal fields
  const { published_at, ...model } = found.model
  send(res, 200, {
    ...model,
    description: `<p>A <b>${model.name}</b> model for testing.</p><p>Works best at 30 steps.<br>Have fun!</p>`,
    versions: versionsFor(found.model, found.family, found.resource, browsing),
  })
})

const downloadWorkers = 2
let downloadsRunning = 0

function pumpDownloads() {
  while (downloadsRunning < downloadWorkers) {
    const next = [...jobs.values()].find((j) => j.status === 'queued' && j.lane === 'io')
    if (!next) return
    downloadsRunning++
    next.status = 'running'
    next.started_at = new Date().toISOString()
    const started = Date.now()
    const total = next.download.size_bytes
    const timer = setInterval(() => {
      if (next.status === 'cancelled') {
        // Cancellable: the transfer stops right away.
        clearInterval(timer)
        next.completed_at = new Date().toISOString()
        next.error = 'Cancelled'
        downloadsRunning--
        return pumpDownloads()
      }
      const progress = Math.min(1, (Date.now() - started) / next.durationMs)
      const done = Math.round(total * progress)
      const elapsed = Math.max((Date.now() - started) / 1000, 0.001)
      next.progress = progress
      next.detail = {
        phase: progress < 1 ? 'downloading' : 'verifying',
        filename: next.download.filename,
        bytes_done: done,
        bytes_total: total,
        speed_bps: Math.round(done / elapsed),
        eta_s: progress < 1 ? Math.round((next.durationMs - (Date.now() - started)) / 1000) : null,
      }
      if (progress < 1) return
      clearInterval(timer)
      try {
        next.result = next.run()
        next.status = 'completed'
      } catch (e) {
        next.status = 'failed'
        next.error = e.message
      }
      next.completed_at = new Date().toISOString()
      downloadsRunning--
      pumpDownloads()
    }, 100)
  }
}

route('POST', '/api/v1/civitai/downloads', (req, res) => {
  const { version_id, file_id, model_type, resource_type, accept_mismatch } = req.body
  const found = findFile(version_id)
  const version = found?.versions.find((v) => v.id === version_id)
  const file = version?.files.find((f) => f.id === file_id)
  if (!file) return send(res, 404, { detail: `File ${file_id} is not part of Civitai version ${version_id}` })
  if (!file.compatible) return send(res, 400, { detail: file.reason })
  if (file.target_resource_type !== resource_type)
    return send(res, 400, { detail: `This file belongs in ${file.target_resource_type}, not ${resource_type}` })
  if (version.model_type !== model_type && !accept_mismatch)
    return send(res, 409, {
      detail: {
        message: "Base model doesn't match; resend with accept_mismatch to download anyway",
        warnings: [`Made for ${version.base_model} (${version.model_type}), not ${model_type}`],
      },
    })
  const active = [...jobs.values()].find(
    (j) => j.lane === 'io' && (j.status === 'queued' || j.status === 'running') && j.download.file_id === file_id,
  )
  if (active)
    return send(res, 409, { detail: { message: 'This file is already being downloaded', job_id: active.job_id } })
  if (models[model_type][resource_type].includes(file.name))
    return send(res, 409, {
      detail: { message: 'Already downloaded', path: `models/${resource_type}/${model_type}/civitai/${file.name}` },
    })

  const path = `models/${resource_type}/${model_type}/civitai/${file.name}`
  const job = submit('civitai_download', 2500, () => {
    if (file.name.includes('broken')) throw new Error('Hash mismatch: download discarded (mock failure)')
    models[model_type][resource_type].push(file.name)
    civitai[file.name] = {
      model_id: found.model.id,
      version_id,
      model_name: found.model.name,
      version_name: version.name,
      base_model: version.base_model,
      trained_words: version.trained_words,
      nsfw: found.model.nsfw,
      url: `https://civitai.com/models/${found.model.id}?modelVersionId=${version_id}`,
    }
    return { path, filename: file.name, model_type, resource_type, sha256: file.sha256, size_bytes: file.size_bytes }
  })
  Object.assign(job, {
    lane: 'io',
    cancellable: true,
    // Transfer time follows file size (~2 s per GB) regardless of MOCK_SPEED, so progress stays visible.
    durationMs: Math.max(300, (file.size_bytes / 2 ** 30) * 2000),
    download: { file_id, filename: file.name, size_bytes: file.size_bytes },
  })
  queueMicrotask(pumpDownloads)
  send(res, 202, {
    job_id: job.job_id,
    filename: file.name,
    path,
    warnings: version.model_type !== model_type ? [`Made for ${version.base_model}`] : [],
  })
})

route('GET', '/api/v1/civitai/files/layout', (req, res) => {
  const q = query(req)
  const versionId = Number(q.get('version_id'))
  const found = findFile(versionId)
  const version = found?.versions.find((v) => v.id === versionId)
  const file = version?.files.find((f) => f.id === Number(q.get('file_id')))
  if (!file) return send(res, 404, { detail: 'File not found' })
  if (version.model_type !== 'flux')
    return send(res, 400, { detail: 'Layout checks are only available for Flux models' })
  if (file.name.includes('locked')) return send(res, 401, { detail: 'Civitai needs an API key for this file' })
  const bundled = version.name.startsWith('v2')
  send(res, 200, {
    layout: { transformer: true, clip_l: bundled, t5xxl: bundled, vae: bundled, format: 'comfy', variant: 'dev' },
    missing_components: bundled ? [] : ['t5xxl'],
  })
})

route('POST', '/api/v1/civitai/identify', (req, res) => {
  const { model_type, resource_type, model_name } = req.body
  if (!models[model_type]?.[resource_type]?.includes(model_name)) return send(res, 404, { detail: 'Model not found' })
  // Pretend hashing takes a moment; only "deliberate" checkpoints are on Civitai.
  setTimeout(() => {
    if (!model_name.includes('deliberate')) return send(res, 200, { matched: false })
    civitai[model_name] = {
      model_id: 4823,
      version_id: 156110,
      model_name: 'Deliberate',
      version_name: 'v3',
      base_model: 'SD 1.5',
      trained_words: [],
      nsfw: false,
      url: 'https://civitai.com/models/4823?modelVersionId=156110',
    }
    send(res, 200, { matched: true, sidecar: { source: 'civitai', ...civitai[model_name] } })
  }, 600 / SPEED)
})

route('GET', '/api/v1/__mock/civitai-image/:size/:id', (_req, res, { id }) => {
  res.writeHead(200, { 'content-type': 'image/png' })
  res.end(Buffer.from(renderImage(150, 200, Number.parseInt(id, 10), `civitai ${id}`), 'base64'))
})

// Test-only: restore the initial state so each end-to-end test is independent.
let seedCount = 30
route('POST', '/__mock/reset', (_req, res) => {
  // Running downloads are cancellable: cancelling stops them before they add models.
  for (const job of jobs.values())
    if (job.status === 'queued' || (job.status === 'running' && job.lane === 'io')) job.status = 'cancelled'
  jobs.clear()
  outputs.length = 0
  seedOutputs(seedCount)
  for (const family of FAMILIES) models[family] = structuredClone(initialModels[family])
  for (const name of Object.keys(civitai)) if (!(name in initialCivitai)) delete civitai[name]
  send(res, 200, { ok: true })
})

// ---------------------------------------------------------------- server
function send(res, status, body) {
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(JSON.stringify(body))
}

/** Seed the gallery so it isn't empty on first load. */
function seedOutputs(count) {
  const prompts = ['a lighthouse at dusk', 'portrait of an astronaut', 'misty pine forest', 'neon city street']
  for (let i = 0; i < count; i++) {
    const [w, h] = [
      [512, 768],
      [768, 512],
      [512, 512],
    ][i % 3]
    const prompt = prompts[i % prompts.length]
    const seed = 1000 + i
    const image = renderImage(w, h, seed, prompt)
    outputs.push({
      id: randomUUID(),
      created_at: new Date(Date.now() - i * 60_000).toISOString(),
      png: Buffer.from(image, 'base64'),
      metadata: {
        generation: {
          instances: [
            {
              prompt,
              negative_prompt: 'blurry',
              num_inference_steps: 30,
              width: w,
              height: h,
              guidance_scale: 7,
              seed,
              scheduler: 'DPM++ 2M',
            },
          ],
        },
        loras: {},
        vae: null,
      },
    })
  }
}

export function startMockApi(port = Number(process.env.MOCK_PORT ?? 8010), { seed = 30 } = {}) {
  seedCount = seed
  seedOutputs(seed)
  const server = http.createServer((req, res) => {
    const path = new URL(req.url, 'http://x').pathname
    const match = routes
      .filter((r) => r.method === req.method)
      .map((r) => ({ r, m: path.match(r.re) }))
      .find(({ m }) => m)
    if (!match) return send(res, 404, { detail: 'Not Found' })
    let data = ''
    req.on('data', (c) => (data += c))
    req.on('end', () => {
      try {
        req.body = data ? JSON.parse(data) : {}
        match.r.handler(req, res, match.m.groups ?? {})
      } catch (e) {
        send(res, 500, { detail: String(e) })
      }
    })
  })
  return new Promise((resolve) => server.listen(port, () => resolve(server)))
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await startMockApi()
  console.log(`Mock sd-api-backend on http://localhost:${process.env.MOCK_PORT ?? 8010} (speed ${SPEED})`)
}
