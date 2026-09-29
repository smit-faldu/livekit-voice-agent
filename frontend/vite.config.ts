import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Forward /api/* to the FastAPI backend, so the browser sees one origin (no CORS).
    proxy: { '/api': 'http://localhost:3000' },
  },
})
