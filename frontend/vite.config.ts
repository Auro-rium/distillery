/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { mockApi } from "./mock/plugin";

export default defineConfig(({ mode }) => {
  const mock = loadEnv(mode, process.cwd(), "VITE_").VITE_MOCK === "1" || process.env.VITE_MOCK === "1";
  return {
    plugins: [react(), ...(mock ? [mockApi()] : [])],
    server: mock ? {} : { proxy: { "/api": "http://localhost:8000" } },
    test: { environment: "jsdom", globals: true },
  };
});
