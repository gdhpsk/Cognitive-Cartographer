'use client';

import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Button, buttonVariants } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Slider } from '@/components/ui/slider';
import { cn } from '@/lib/utils';

type HeadHeatmap = {
  size: number;
  values: Float32Array;
};

const TILE_PX = 104;

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function createHeadHeatmap(size: number): HeadHeatmap {
  return {
    size,
    values: new Float32Array(size * size),
  };
}

function ensureHeatmapGrid(grid: HeadHeatmap[][], layers: number, heads: number, seqLen: number): void {
  const safeLayers = Math.max(0, layers);
  const safeHeads = Math.max(0, heads);
  const safeSeqLen = Math.max(1, seqLen);

  while (grid.length < safeLayers) {
    grid.push([]);
  }

  for (let layerIndex = 0; layerIndex < safeLayers; layerIndex++) {
    const row = grid[layerIndex] ?? [];
    grid[layerIndex] = row;

    while (row.length < safeHeads) {
      row.push(createHeadHeatmap(safeSeqLen));
    }

    for (let headIndex = 0; headIndex < safeHeads; headIndex++) {
      const tile = row[headIndex];
      if (!tile || tile.size !== safeSeqLen) {
        row[headIndex] = createHeadHeatmap(safeSeqLen);
      }
    }
  }
}

function paintHeatmap(canvas: HTMLCanvasElement, heatmap: HeadHeatmap | null): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  if (!heatmap) {
    canvas.width = 1;
    canvas.height = 1;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 1, 1);
    return;
  }

  const { size, values } = heatmap;
  if (canvas.width !== size) canvas.width = size;
  if (canvas.height !== size) canvas.height = size;

  const image = ctx.createImageData(size, size);
  const data = image.data;

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const pixelIndex = (row * size + col) * 4;
      const value = col <= row ? clamp01(values[row * size + col]) : 0;
      const intensity = Math.round(value * 255);

      data[pixelIndex] = 0;
      data[pixelIndex + 1] = intensity;
      data[pixelIndex + 2] = intensity;
      data[pixelIndex + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
}

function writeMatrix(heatmap: HeadHeatmap, matrixRows: unknown[]): void {
  const size = heatmap.size;

  for (let row = 0; row < Math.min(size, matrixRows.length); row++) {
    const rowValues = matrixRows[row];
    if (!Array.isArray(rowValues)) continue;

    const maxCol = Math.min(size, rowValues.length, row + 1);
    for (let col = 0; col < maxCol; col++) {
      heatmap.values[row * size + col] = clamp01(Number(rowValues[col]));
    }
  }
}

function writeRow(heatmap: HeadHeatmap, rowValues: unknown[], rowIndex: number): void {
  const size = heatmap.size;
  if (size <= 0) return;

  const safeRow = Math.max(0, Math.min(size - 1, rowIndex));
  const maxCol = Math.min(size, rowValues.length, safeRow + 1);
  for (let col = 0; col < maxCol; col++) {
    heatmap.values[safeRow * size + col] = clamp01(Number(rowValues[col]));
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function toPositiveInt(value: unknown, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.floor(parsed);
}

function HeatmapTile({
  heatmap,
  layerIndex,
  headIndex,
  version,
}: {
  heatmap: HeadHeatmap | null;
  layerIndex: number;
  headIndex: number;
  version: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    paintHeatmap(canvasRef.current, heatmap);
  }, [heatmap, version]);

  const size = heatmap?.size ?? 1;

  return (
    <div className="rounded-md border border-cyan-300/20 bg-black/80 p-2">
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="h-24 w-24 rounded border border-white/10 bg-black [image-rendering:pixelated]"
      />
      <p className="mt-1 text-center text-[10px] text-muted-foreground">
        L{layerIndex + 1} H{headIndex + 1}
      </p>
    </div>
  );
}

export default function AttentionHeatmapPage() {
  const [query, setQuery] = useState('');
  const [k, setK] = useState(5);
  const [maxTokens, setMaxTokens] = useState(200);
  const [isStreaming, setIsStreaming] = useState(false);
  const [statusText, setStatusText] = useState('Idle');

  const [promptTokens, setPromptTokens] = useState<string[]>([]);
  const [generatedTokens, setGeneratedTokens] = useState<string[]>([]);
  const [answerText, setAnswerText] = useState('');

  const [seqLen, setSeqLen] = useState(0);
  const [totalLayers, setTotalLayers] = useState(0);
  const [totalHeads, setTotalHeads] = useState(0);
  const [renderVersion, setRenderVersion] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const heatmapsRef = useRef<HeadHeatmap[][]>([]);
  const seqLenRef = useRef(0);
  const totalLayersRef = useRef(0);
  const totalHeadsRef = useRef(0);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  const stopStreaming = useCallback((message = 'Stopped.') => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsStreaming(false);
    setStatusText(message);
  }, []);

  const handleAttentionMessage = useCallback((raw: unknown) => {
    if (!isRecord(raw)) return;

    const eventName = typeof raw.event === 'string' ? raw.event : '';
    const data = raw.data;

    if (eventName === 'tokens' && isRecord(data)) {
      const incomingTokens = Array.isArray(data.prompt_tokens)
        ? data.prompt_tokens.map((token) => String(token))
        : [];
      const layers = toPositiveInt(data.total_layers, 0);
      const nextSeqLen = Math.max(1, incomingTokens.length);

      seqLenRef.current = nextSeqLen;
      totalLayersRef.current = layers;
      totalHeadsRef.current = 0;
      heatmapsRef.current = Array.from({ length: layers }, () => []);

      setPromptTokens(incomingTokens);
      setGeneratedTokens([]);
      setAnswerText('');
      setSeqLen(nextSeqLen);
      setTotalLayers(layers);
      setTotalHeads(0);
      setStatusText(`Streaming attention for ${incomingTokens.length} prompt tokens...`);
      setRenderVersion((v) => v + 1);
      return;
    }

    if (eventName === 'gen_step' && isRecord(data)) {
      const step = Math.max(0, Math.floor(Number(data.step) || 0));
      const token = typeof data.token === 'string' ? data.token : '';
      if (token) {
        setGeneratedTokens((prev) => [...prev, token]);
      }

      const layerPayloads = Array.isArray(data.layers) ? data.layers : [];
      let nextMaxHeadCount = totalHeadsRef.current;

      for (const layerPayload of layerPayloads) {
        if (!isRecord(layerPayload)) continue;

        const layerIndex = Math.max(0, Math.floor(Number(layerPayload.layer) || 0));
        const headWeightsRaw = Array.isArray(layerPayload.head_weights)
          ? layerPayload.head_weights
          : [];

        if (layerIndex + 1 > totalLayersRef.current) {
          totalLayersRef.current = layerIndex + 1;
        }

        if (headWeightsRaw.length > nextMaxHeadCount) {
          nextMaxHeadCount = headWeightsRaw.length;
        }

        ensureHeatmapGrid(
          heatmapsRef.current,
          totalLayersRef.current,
          nextMaxHeadCount,
          Math.max(1, seqLenRef.current)
        );

        for (let headIndex = 0; headIndex < headWeightsRaw.length; headIndex++) {
          const headData = headWeightsRaw[headIndex];
          const heatmap = heatmapsRef.current[layerIndex]?.[headIndex];
          if (!heatmap || !Array.isArray(headData)) continue;

          if (headData.length > 0 && Array.isArray(headData[0])) {
            writeMatrix(heatmap, headData as unknown[]);
          } else {
            writeRow(heatmap, headData as unknown[], step);
          }
        }
      }

      totalHeadsRef.current = nextMaxHeadCount;
      setTotalLayers(totalLayersRef.current);
      setTotalHeads(nextMaxHeadCount);
      setStatusText(`Streaming step ${step + 1}...`);
      setRenderVersion((v) => v + 1);
      return;
    }

    if (eventName === 'answer' && isRecord(data)) {
      const answer = typeof data.answer === 'string' ? data.answer : '';
      setAnswerText(answer);
      setStatusText('Answer received. Waiting for attention_done...');
      return;
    }

    if (eventName === 'attention_done' && isRecord(data)) {
      const layers = toPositiveInt(data.total_layers, totalLayersRef.current || 0);
      const heads = toPositiveInt(data.total_heads, totalHeadsRef.current || 0);

      totalLayersRef.current = layers;
      totalHeadsRef.current = heads;

      if (layers > 0 && heads > 0) {
        ensureHeatmapGrid(
          heatmapsRef.current,
          layers,
          heads,
          Math.max(1, seqLenRef.current)
        );
      }

      setTotalLayers(layers);
      setTotalHeads(heads);
      setRenderVersion((v) => v + 1);
      stopStreaming(`Attention complete: ${layers} layers x ${heads} heads.`);
      return;
    }
  }, [stopStreaming]);

  const startStreaming = useCallback(() => {
    if (!query.trim() || isStreaming) return;

    const hostname = process.env.NEXT_PUBLIC_HOSTNAME;
    if (!hostname) {
      setStatusText('Missing NEXT_PUBLIC_HOSTNAME.');
      return;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    seqLenRef.current = 0;
    totalLayersRef.current = 0;
    totalHeadsRef.current = 0;
    heatmapsRef.current = [];

    setPromptTokens([]);
    setGeneratedTokens([]);
    setAnswerText('');
    setSeqLen(0);
    setTotalLayers(0);
    setTotalHeads(0);
    setRenderVersion((v) => v + 1);

    setIsStreaming(true);
    setStatusText('Connecting to /ws/attention...');

    const ws = new WebSocket(`wss://${hostname}/ws/attention`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatusText('Connected. Requesting normalized attention heads...');
      ws.send(
        JSON.stringify({
          query: query.trim(),
          k,
          max_tokens: maxTokens,
        })
      );
    };

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data) as unknown;
        handleAttentionMessage(parsed);
      } catch {
        setStatusText('Received malformed websocket payload.');
      }
    };

    ws.onerror = () => {
      setIsStreaming(false);
      setStatusText('WebSocket error while streaming attention.');
    };

    ws.onclose = () => {
      wsRef.current = null;
      setIsStreaming(false);
    };
  }, [handleAttentionMessage, isStreaming, k, maxTokens, query]);

  const renderLayerCount = Math.max(totalLayers, heatmapsRef.current.length);

  return (
    <main className="min-h-screen w-full bg-[#030911] px-4 py-4 text-white sm:px-6">
      <div className="mx-auto flex w-full max-w-475 flex-col gap-4">
        <Card className="border-cyan-300/20 bg-black/60 backdrop-blur-xl">
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <CardTitle className="text-xl text-white">Attention Head Heatmap Viewer</CardTitle>
                <p className="mt-1 text-sm text-muted-foreground">
                  Each tile is one head. Pixel color maps normalized magnitude from black (0) to cyan (1).
                </p>
              </div>
              <Link
                href="/"
                className={cn(buttonVariants({ variant: 'secondary' }))}
              >
                Back to Graph
              </Link>
            </div>
          </CardHeader>
          <CardContent>
            <form
              className="flex flex-col gap-3"
              onSubmit={(e) => {
                e.preventDefault();
                startStreaming();
              }}
            >
              <div className="flex flex-wrap items-end gap-3">
                <Input
                  placeholder="Ask a question for /ws/attention..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="min-w-70 flex-1 bg-black/50 text-white"
                />
                <div className="w-40">
                  <p className="mb-1 text-xs text-muted-foreground">k: {k}</p>
                  <Slider
                    min={1}
                    max={20}
                    step={1}
                    value={[k]}
                    onValueChange={(value) => {
                      const next = Array.isArray(value) ? value[0] : value;
                      if (typeof next === 'number') setK(next);
                    }}
                  />
                </div>
                <div className="w-32">
                  <p className="mb-1 text-xs text-muted-foreground">max_tokens</p>
                  <Input
                    type="number"
                    min={1}
                    max={2048}
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(toPositiveInt(e.target.value, 200))}
                    className="bg-black/50 text-white"
                  />
                </div>
                <Button type="submit" disabled={isStreaming || !query.trim()}>
                  {isStreaming ? 'Streaming...' : 'Start Attention Stream'}
                </Button>
                <Button
                  type="button"
                  variant="secondary"
                  disabled={!isStreaming}
                  onClick={() => stopStreaming()}
                >
                  Stop
                </Button>
              </div>
            </form>

            <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
              <span>Status: {statusText}</span>
              <span>seq_len: {seqLen || '-'}</span>
              <span>layers: {totalLayers || '-'}</span>
              <span>heads: {totalHeads || '-'}</span>
            </div>

            <div className="mt-3 w-64">
              <div className="h-2 rounded-full bg-linear-to-r from-black to-cyan-400" />
              <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
                <span>0.0</span>
                <span>1.0</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="border-white/10 bg-black/50">
            <CardHeader>
              <CardTitle className="text-sm text-white">Prompt Tokens ({promptTokens.length})</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-28 rounded border border-white/10 bg-black/40 p-2">
                <div className="flex flex-wrap gap-1.5">
                  {promptTokens.length === 0 ? (
                    <p className="text-xs text-muted-foreground">No tokens yet.</p>
                  ) : (
                    promptTokens.map((token, index) => (
                      <span
                        key={`${token}-${index}`}
                        className="rounded border border-cyan-400/25 bg-cyan-500/10 px-1.5 py-0.5 font-mono text-[11px]"
                      >
                        {token}
                      </span>
                    ))
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          <Card className="border-white/10 bg-black/50">
            <CardHeader>
              <CardTitle className="text-sm text-white">
                Generated Tokens ({generatedTokens.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-28 rounded border border-white/10 bg-black/40 p-2">
                {generatedTokens.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No generated tokens yet.</p>
                ) : (
                  <p className="font-mono text-xs leading-relaxed text-white/85">
                    {generatedTokens.join('')}
                  </p>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        <Card className="border-cyan-300/20 bg-black/50">
          <CardHeader>
            <CardTitle className="text-sm text-white">Layer x Head Grid</CardTitle>
          </CardHeader>
          <CardContent>
            {renderLayerCount === 0 || totalHeads === 0 ? (
              <p className="text-sm text-muted-foreground">
                Start a stream to render per-head seq_len x seq_len heatmaps.
              </p>
            ) : (
              <div className="max-h-[68vh] overflow-auto rounded-md border border-white/10 bg-black/70 p-3">
                <div
                  className="grid gap-2"
                  style={{
                    gridTemplateColumns: `repeat(${totalHeads}, ${TILE_PX}px)`,
                  }}
                >
                  {Array.from({ length: renderLayerCount }).map((_, layerIndex) =>
                    Array.from({ length: totalHeads }).map((__, headIndex) => (
                      <HeatmapTile
                        key={`layer-${layerIndex}-head-${headIndex}`}
                        heatmap={heatmapsRef.current[layerIndex]?.[headIndex] ?? null}
                        layerIndex={layerIndex}
                        headIndex={headIndex}
                        version={renderVersion}
                      />
                    ))
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-white/10 bg-black/50">
          <CardHeader>
            <CardTitle className="text-sm text-white">Final Answer</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-32 rounded border border-white/10 bg-black/40 p-2">
              {answerText ? (
                <p className="text-sm leading-relaxed text-white/90">{answerText}</p>
              ) : (
                <p className="text-sm text-muted-foreground">No answer received yet.</p>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
