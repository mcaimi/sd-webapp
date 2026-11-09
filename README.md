# SD Studio

A web interface for Stable Diffusion image generation, inpainting and model merging. All inference runs on the
[sd-api-backend](../sd-api-backend) REST service; this repository is the frontend, a single-page app with no GPU
or model files of its own.

![app](assets/app.png)

## Features

- **Model families:** SD 1.5, SDXL, Illustrious XL and Flux, each with sensible defaults, native-resolution presets
  and its own settings (e.g. Flux's True CFG and FlowMatch schedulers).
- **Generate:** batches of up to 8, side-by-side checkpoint comparisons on the same seed, LoRAs with strength
  sliders, custom VAEs, seed locking, and a prompt token counter. Any result's settings can be loaded back.
- **Inpaint:** drop, browse or paste an image and paint the mask with brush and eraser, undo/redo and touch
  support. A before/after slider compares the result with the original.
- **Merge:** linear, SLERP, additive and subtractive merges, in single, batch or multi-step recipe mode, with
  downloads and one click to use the new checkpoint.
- **Gallery:** browse everything the backend has saved, view the generation details, remix an image into a new
  generation, and delete with undo.
- **Background jobs:** keep working while jobs queue and run. A jobs panel and toasts track them, with optional
  browser notifications. Jobs and results survive reloads.
- **System:** live GPU, memory and CPU usage with short-term history.
- Responsive from phone to desktop, light and dark themes, keyboard accessible (command palette on ⌘/Ctrl K), and
  checked against WCAG 2.1 AA in the end-to-end tests.

## Run locally

With [sd-api-backend](../sd-api-backend) running on `http://localhost:8000`:

```sh
cd web
npm install
npm run dev                                   # http://localhost:5173
SD_API_URL=http://gpu-host:8000 npm run dev   # backend elsewhere
```

No GPU at hand? `npm run dev:mock` runs the UI against a built-in mock of the backend.

See [web/README.md](web/README.md) for the stack, scripts, project layout and testing.

## Deploy

`web/Dockerfile` builds an unprivileged nginx image (port 8080, OpenShift-compatible). It serves the app and
proxies `/api` to the backend named by `SD_API_URL`, so the browser only ever talks to one origin.

```sh
docker build -t sd-webapp-web web
docker run -p 8080:8080 -e SD_API_URL=http://host.docker.internal:8000 sd-webapp-web
```

Kubernetes/OpenShift manifests (Deployment, Service, Route) live in `web/deploy`:

```sh
cd web/deploy && kustomize edit set image sd-webapp-web=quay.io/<org>/sd-webapp-web:<tag>
oc apply -k web/deploy
```

## History

Up to commit `14a4628` this was a Streamlit application, first running inference locally and later calling the
backend. It was replaced by the web app in `web/`.
