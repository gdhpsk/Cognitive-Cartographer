# Cognitive Cartographer

A 3D knowledge graph visualizer with real-time attention head visualization, AI-powered Q&A, and GPT-4o heatmap analysis. Upload a PDF, explore its latent space as an interactive force-directed graph, query it with natural language, and inspect transformer attention patterns at every layer and head.

## Features

### 3D Knowledge Graph
- Force-directed 3D graph layout powered by **d3-force-3d** and rendered with **React Three Fiber**
- Upload a PDF via WebSocket and the backend extracts nodes and edges into an explorable graph
- Click nodes to inspect content, connections, and metadata in a slide-out detail panel
- Search nodes by label or content with real-time highlighting
- Camera animates to center on selected nodes with orbit controls locked to origin

### AI Question & Answer
- Ask natural language questions against the uploaded document via WebSocket (`/ws/attention`)
- Responses include source citations linked to graph nodes — click a source to fly to the relevant node
- Model selection with live switching via `/choose_llm` and `/aval_model` API endpoints
- Configurable retrieval parameter `k` (number of context chunks) and max tokens
- Queue-aware: when the server is at its inference concurrency limit, a queue-position banner with an animated pulse is shown so users know to wait — the graph is still fully explorable in the meantime
- Session-based: each upload creates a persistent server-side session tied to the upload WebSocket lifetime

### Attention Head Visualization
- Real-time streaming of transformer attention weights during inference
- Three view modes:
  - **3D Stack** — all layers rendered as stacked planes with CSS 3D transforms; drag to pan, scroll/pinch to zoom
  - **Layer Overview** — grid of all heads for a single layer; click a head to drill down
  - **Single Head** — full-resolution heatmap for one layer/head pair
- WebGL2 heatmap renderer with GLSL shaders for performant rendering of large attention matrices
- Canvas fallback for browsers without WebGL2 support
- Hover tooltips showing token pair and attention intensity
- Global or per-head scale normalization toggle
- Collapsible token strips showing prompt and generated token sequences

### GPT-4o Heatmap Analysis
- Capture the current heatmap view as a PNG and send it to GPT-4o via `/image_analysis`
- The request is enriched with full context: token list, view mode, layer/head selection, heatmap statistics
- Analysis response rendered as markdown
- Customizable analysis prompt

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 16 (App Router, Turbopack) |
| 3D Rendering | React Three Fiber, @react-three/drei, Three.js |
| Graph Physics | d3-force-3d |
| State Management | Zustand v5 |
| UI Components | shadcn/ui (base-ui primitives), Aceternity UI patterns |
| Animations | Motion (Framer Motion), GSAP |
| Styling | Tailwind CSS v4 |
| Heatmap Rendering | WebGL2 with custom GLSL shaders |
| Markdown | react-markdown |
| Font | Oxygen / Oxygen Mono (Google Fonts) |

## Project Structure

```
app/
  page.tsx          # Main application — UI, WebSocket handlers, attention visualization
  layout.tsx        # Root layout with fonts and dark mode
  globals.css       # Tailwind config, CSS variables, custom animations
components/
  scene.tsx         # React Three Fiber scene — ForceGraph, CameraController, edges, tooltips
  file-upload.tsx   # Aceternity-style drag-and-drop PDF upload
  ui/               # shadcn + custom UI components
    spotlight-card.tsx   # Cursor-tracking spotlight glow effect
    text-generate.tsx    # Staggered word reveal animation
    glowing-border.tsx   # Animated conic-gradient border
    button.tsx, input.tsx, card.tsx, slider.tsx, scroll-area.tsx,
    separator.tsx, switch.tsx, tabs.tsx, select.tsx, textarea.tsx, badge.tsx
helpers/
  store.ts          # Zustand store — nodes, edges, selection, active/AI-source state
types/
  d3-force-3d.d.ts  # Type declarations for d3-force-3d
```

## Getting Started

### Prerequisites

- Node.js 20+ or Bun
- A running backend. See the [backend branch](https://github.com/gdhpsk/Cognitive-Cartographer/tree/backend) for setup instructions. It must provide the following endpoints:
  - `GET /graph` — returns `{ nodes, edges }`
  - `GET /aval_model` — returns `{ available_models: string[] }`
  - `PATCH /choose_llm` — switches the active LLM
  - `POST /image_analysis` — accepts `image` (file) + `prompt` (form field), returns `{ analysis: string }`
  - `WSS /ws/upload` — accepts PDF bytes, streams progress events, returns `done` with `session_id`. **Keep this socket open** — closing it destroys the server-side session.
  - `GET /graph?session_id=<uuid>` — returns `{ nodes, edges }` for the active session
  - `WSS /ws/attention` — accepts `{ query, k, max_tokens, session_id }`, streams `graph` → (optional `queued`) → `tokens` → `attention`/`gen_step` → `answer`/`attention_done`

### Connecting to the Backend

On first launch, the app displays a setup screen prompting for your backend API hostname (e.g. `api.example.com`). Enter the hostname only — HTTPS and WSS protocols are applied automatically. The value is saved to `localStorage` so you only need to enter it once. You can change it anytime via the "change" link in the header card.

### Environment Variables

No frontend environment variables are required.

### Install and Run

```bash
# Install dependencies
bun install

# Start development server
bun dev

# Production build
bun run build
bun start
```

Open [http://localhost:3000](http://localhost:3000). Minimum screen width: 1200px.

## WebSocket Protocol

### `/ws/attention`

**Send:** `{ "query": "...", "k": 10, "max_tokens": 200, "session_id": "<uuid>" }`

The `session_id` is required and obtained from the `/ws/upload` `done` event. The session lives as long as the upload WebSocket remains connected.

**Receive (event stream):**

| Event | Payload | Description |
|-------|---------|-------------|
| `graph` | `{ nodes, edges, path }` | Knowledge graph with highlighted source path — always sent first |
| `queued` | `{ position, message }` | Server concurrency limit reached; request is queued at `position`. The UI shows a queue-position banner. Omitted on fast-path. |
| `tokens` | `{ prompt_tokens, seq_len, total_layers, total_heads }` | Inference started (clears queued state); initializes the attention grid |
| `attention` | `{ step, token, attention_grid, … }` | Full attention matrices for all layers/heads at a given step |
| `gen_step` | `{ step, token, layers: [{ layer, head_weights }] }` | Incremental attention row per generation step |
| `answer` | `{ answer, sources }` | Final AI response with source citations |
| `attention_done` | `{ total_layers, total_heads }` | Signals completion of attention streaming |
| `error` | `string \| { message }` | Terminal error (e.g. invalid/expired session). Input is re-enabled. |

## License

MIT
