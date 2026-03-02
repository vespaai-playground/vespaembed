import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../src/vespaembed/static/dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/upload': 'http://127.0.0.1:8000',
      '/train': 'http://127.0.0.1:8000',
      '/stop': 'http://127.0.0.1:8000',
      '/runs': 'http://127.0.0.1:8000',
      '/active_run_id': 'http://127.0.0.1:8000',
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
