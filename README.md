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
- Configurable retrieval parameter `k` (number of context chunks)

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
  - `WSS /ws/upload` — accepts PDF bytes, streams progress, returns `done`
  - `WSS /ws/attention` — accepts `{ query, k }`, streams `tokens`, `attention`/`gen_step`, `answer`/`attention_done` events

### Connecting to the Backend

On first launch, the app displays a setup screen prompting for your backend API hostname (e.g. `graph.example.com`). Enter the hostname only — HTTPS and WSS protocols are applied automatically. The value is saved to `localStorage` so you only need to enter it once. You can change it anytime via the "change" link in the header card.

### Environment Variables (optional)

```env
NEXT_PUBLIC_ALLOW_CUSTOM=off
```

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_ALLOW_CUSTOM` | Set to `on` to allow entering custom model names in the model selector. |

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

**Send:** `{ "query": "...", "k": 10 }`

**Receive (event stream):**

| Event | Description |
|-------|-------------|
| `tokens` | `{ prompt_tokens, seq_len, total_layers, total_heads }` — initializes the attention grid |
| `attention` | Full attention matrices for all layers/heads at a given step |
| `gen_step` | Incremental attention row per generation step with `{ step, token, layers: [{ layer, head_weights }] }` |
| `graph` | `{ nodes, edges, path }` — the knowledge graph with highlighted source path |
| `answer` | `{ answer, sources }` — the final AI response with source citations |
| `attention_done` | Signals completion of attention streaming |

## License

MIT
