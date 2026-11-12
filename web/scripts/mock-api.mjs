// Mock sd-api-backend for UI development and end-to-end tests.
//
// Mirrors the real backend's behaviour where the UI depends on it: a single
// sequential worker, jobs going queued -> running -> completed/failed/cancelled,
// progress that jumps from 0 to 1 on completion, the same result shapes, and
// persisted outputs with the backend's metadata layout. Images are generated
// gradients encoded as real PNGs.
//
// Knobs:  MOCK_PORT (8010), MOCK_SPEED (1 = realistic-ish, 0.2 = fast).
// A prompt containing "fail" makes the job fail; "slow" makes it 5x slower.
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

// Snapshot so tests can reset merges back to the original model lists.
const initialModels = structuredClone(models)

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
    status: 'queued',
    progress: 0,
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
  const next = [...jobs.values()].find((j) => j.status === 'queued')
  if (!next) return
  running = next
  next.status = 'running'
  next.started_at = new Date().toISOString()
  setTimeout(() => {
    // Like the real worker: cancelling a running job doesn't stop it, and
    // completion overwrites the "cancelled" status.
    try {
      next.result = next.run()
      next.status = 'completed'
      next.progress = 1
    } catch (e) {
      next.status = 'failed'
      next.error = e.message
    }
    next.completed_at = new Date().toISOString()
    running = null
    pump()
  }, next.durationMs)
}
// eslint-disable-next-line no-unused-vars -- strip internal fields
const publicJob = ({ durationMs, run, ...job }) => job

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
    cache_stats: { cached_models: 1, max_models: 2, cache_keys: ['sdxl:juggernautXL_v9'] },
    queue_stats: {
      queue_size: all.filter((j) => j.status === 'queued').length,
      total_jobs: all.length,
      num_workers: 1,
      workers_running: true,
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
route('GET', '/api/v1/info/schedulers', (_req, res) => send(res, 200, SD_SCHEDULERS))

route('GET', '/api/v1/models/:family/:resource', (_req, res, { family, resource }) => {
  if (!models[family]?.[resource]) return send(res, 400, { detail: 'Invalid model type or resource' })
  send(res, 200, {
    model_type: family,
    resource_type: resource,
    models: models[family][resource].map((name) => ({ name, path: `models/${resource}/${family}/${name}` })),
  })
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
  if (!job || job.status === 'completed' || job.status === 'failed')
    return send(res, 400, { detail: 'Job cannot be cancelled (already completed or failed)' })
  job.status = 'cancelled'
  job.completed_at = new Date().toISOString()
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

// Test-only: restore the initial state so each end-to-end test is independent.
let seedCount = 30
route('POST', '/__mock/reset', (_req, res) => {
  for (const job of jobs.values()) if (job.status === 'queued') job.status = 'cancelled'
  jobs.clear()
  outputs.length = 0
  seedOutputs(seedCount)
  for (const family of FAMILIES) models[family] = structuredClone(initialModels[family])
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
