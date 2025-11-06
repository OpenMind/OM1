import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Correct Vite config for root-level React app
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
  },
});
