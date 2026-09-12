/** @type {import('next').NextConfig} */
const API = process.env.RAG_API_URL ?? "http://127.0.0.1:8000";
const isDev = process.env.NODE_ENV !== "production";

const nextConfig = {
  // Progress lines and answer tokens must reach the browser as they are
  // produced. Next's gzip compression buffers a response until it has enough
  // to compress, which held the whole NDJSON stream back until it ended.
  compress: false,
  ...(isDev
    ? {
        experimental: {
          // The rewrite proxy gives up after 30 s by default. Indexing a scanned
          // folder, or waiting for a CPU model's first token, routinely takes
          // longer, and the browser was cut off mid-request with no answer.
          proxyTimeout: 30 * 60 * 1000,
        },
        // In development, proxy the FastAPI backend through Next so the browser
        // only ever talks to one origin: no CORS.
        async rewrites() {
          return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
        },
      }
    : {
        // `next build` emits a static site (the page is entirely client-side),
        // which FastAPI serves itself at :8000: one process, no proxy at all.
        output: "export",
      }),
};

export default nextConfig;
