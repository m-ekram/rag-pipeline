/** @type {import('next').NextConfig} */
const API = process.env.RAG_API_URL ?? "http://127.0.0.1:8000";

const nextConfig = {
  // Proxy the FastAPI backend through Next so the browser only ever talks to
  // one origin. Avoids CORS entirely, and keeps streaming responses intact —
  // rewrites pass the body through without buffering, which SSE/NDJSON needs.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
