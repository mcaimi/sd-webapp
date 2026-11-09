import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  // The SPA always calls the API on its own origin (/api/...). In dev, Vite
  // proxies those calls to the backend; in production a reverse proxy does.
  const apiTarget = env.SD_API_URL || 'http://localhost:8000'

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: { '@': path.resolve(import.meta.dirname, 'src') },
    },
    server: {
      proxy: { '/api': { target: apiTarget, changeOrigin: true } },
    },
    build: {
      rolldownOptions: {
        output: {
          // Long-lived vendor chunks: app releases don't invalidate cached libraries.
          codeSplitting: {
            groups: [
              { name: 'react', test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/ },
              { name: 'router', test: /node_modules[\\/]react-router[\\/]/ },
              { name: 'motion', test: /node_modules[\\/](motion|motion-dom|motion-utils|framer-motion)[\\/]/ },
              { name: 'query', test: /node_modules[\\/]@tanstack[\\/]/ },
              { name: 'ui', test: /node_modules[\\/](@radix-ui|radix-ui|@floating-ui|cmdk|sonner|lucide-react)[\\/]/ },
            ],
          },
        },
      },
    },
    preview: {
      proxy: { '/api': { target: apiTarget, changeOrigin: true } },
    },
  }
})
