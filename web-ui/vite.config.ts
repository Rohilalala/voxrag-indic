import path from 'node:path'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, type PluginOption } from 'vite'

/** FastAPI mounts the repo's `web/` directory at `/static`, so the built bundle lives at
 *  `/static/dist/`. `base` has to match or every asset URL 404s in production. */
const BASE = '/static/dist/'

const BACKEND = 'http://127.0.0.1:8000'

/** Dev-only convenience: `base` also prefixes the dev server, so a bare `/` would 404.
 *  Send it to the app instead of making everyone remember the path. */
function redirectRootToBase(): PluginOption {
  return {
    name: 'voxrag-dev-root-redirect',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url === '/' || req.url === '/index.html') {
          res.writeHead(302, { Location: BASE })
          res.end()
          return
        }
        next()
      })
    },
  }
}

export default defineConfig({
  base: BASE,
  plugins: [react(), tailwindcss(), redirectRootToBase()],
  resolve: {
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  build: {
    outDir: '../web/dist',
    emptyOutDir: true,
    // The page is opened once; a couple of extra KB beats a chunk-size warning wall.
    chunkSizeWarningLimit: 1200,
    // Everything must be bundled -- no runtime CDN requests, no external font fetch.
    assetsInlineLimit: 4096,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/ws': { target: BACKEND, changeOrigin: true, ws: true },
    },
  },
})
