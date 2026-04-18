'use client'
import { useState, useRef, useCallback, useEffect } from 'react';
import Scene from '@/components/scene';
import { useAppStore, generateRandomHex, type GraphData } from '@/helpers/store';
import { FileUpload } from '@/components/file-upload';
import { Slider } from '@/components/slider';

const MIN_SCREEN_WIDTH = 1200;

interface QuerySource {
  text: string;
  metadata: { chat_id: number; length: number; source: string };
}

export default function App() {
  const [queryText, setQueryText] = useState('');
  const [aiQuery, setAiQuery] = useState('');
  const [aiK, setAiK] = useState(10);
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [aiSources, setAiSources] = useState<QuerySource[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [hasUploadedPdf, setHasUploadedPdf] = useState(false);
  const [pendingUploadFile, setPendingUploadFile] = useState<File | null>(null);
  const [isScreenWideEnough, setIsScreenWideEnough] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);
  const { nodes, edges, isLoading, setLoading, setActiveNodes, setAiSourceNodes, selectedNodeId, setSelectedNode, loadGraph } = useAppStore();

  const handleAiQuery = useCallback(() => {
    if (!aiQuery.trim() || isQuerying) return;
    setIsQuerying(true);
    setAiAnswer(null);
    setAiSources([]);

    const ws = new WebSocket(`wss://${process.env.NEXT_PUBLIC_HOSTNAME}/ws/query`);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ query: aiQuery.trim(), k: aiK }));
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.event === 'answer') {
        setAiAnswer(msg.data.answer);
        const sources: QuerySource[] = msg.data.sources || [];
        setAiSources(sources);
        setIsQuerying(false);
        ws.close();

        // Match sources to graph nodes and activate them
        const { nodes: currentNodes, edges: currentEdges } = useAppStore.getState();
        const sourceNodeIds = new Set<string>();
        for (const src of sources) {
          for (const node of currentNodes) {
            if (node.text === src.text) {
              sourceNodeIds.add(node.id);
              break;
            }
          }
        }
        const matchedIds = new Set<string>(sourceNodeIds);
        // Include direct neighbors
        for (const edge of currentEdges) {
          if (matchedIds.has(edge.sourceId)) matchedIds.add(edge.targetId);
          if (matchedIds.has(edge.targetId)) matchedIds.add(edge.sourceId);
        }
        // Keep the full AI-answer cluster (sources + neighbors) pinned as active.
        useAppStore.getState().setAiSourceNodes([...matchedIds]);
        useAppStore.getState().setActiveNodes([...matchedIds]);
      }
    };

    ws.onerror = () => {
      setAiAnswer('Failed to connect to query service.');
      setIsQuerying(false);
    };

    ws.onclose = () => {
      wsRef.current = null;
    };
  }, [aiK, aiQuery, isQuerying]);

  const API_BASE = `https://${process.env.NEXT_PUBLIC_HOSTNAME}`;

  const handleLoadGraph = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/graph`);
      const data: GraphData = await res.json();
      const graphNodes = data.nodes.map((n) => {
        return {
          id: n.id,
          label: n.label,
          text: n.text || '',
          position: [n.x, n.y, n.z] as [number, number, number],
          color: generateRandomHex(),
        };
      });
      const graphEdges = data.edges.map((e) => ({
        sourceId: e.source,
        targetId: e.target,
      }));
      loadGraph(graphNodes, graphEdges);
    } catch (err) {
      console.error('Failed to load graph:', err);
    } finally {
      setLoading(false);
    }
  }, [loadGraph, setLoading]);

  const handleUploadFileSelection = useCallback((files: File[]) => {
    const file = files?.[0];
    if (!file || isUploading) return;

    setHasUploadedPdf(false);
    setPendingUploadFile(file);
    setUploadStatus(`Ready to upload: ${file.name}`);
  }, [isUploading]);

  const handleConfirmUpload = useCallback(() => {
    if (!pendingUploadFile || isUploading) return;

    setHasUploadedPdf(false);
    setIsUploading(true);
    setUploadStatus('Connecting to upload service...');

    const uploadWs = new WebSocket(`wss://${process.env.NEXT_PUBLIC_HOSTNAME}/ws/upload`);

    uploadWs.onopen = () => {
      setUploadStatus('Connected. Uploading PDF bytes...');
      uploadWs.send(pendingUploadFile);
    };

    uploadWs.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data);
        if (typeof parsed?.data === 'string') {
          setUploadStatus(parsed.data);
        }
        if (parsed?.event === 'done') {
          setHasUploadedPdf(true);
          setPendingUploadFile(null);
          setIsUploading(false);
          void handleLoadGraph();
          uploadWs.close();
        }
      } catch {
        // Keep the raw message as a fallback status for non-JSON responses.
        if (typeof msg.data === 'string') {
          setUploadStatus(msg.data);
        }
      }
    };

    uploadWs.onerror = () => {
      setUploadStatus('Upload failed. Check that ws://localhost:8000 is running.');
      setHasUploadedPdf(false);
      setIsUploading(false);
    };

    uploadWs.onclose = () => {
      setIsUploading(false);
    };
  }, [handleLoadGraph, isUploading, pendingUploadFile]);

  useEffect(() => {
    const checkViewportWidth = () => {
      setIsScreenWideEnough(window.innerWidth >= MIN_SCREEN_WIDTH);
    };

    checkViewportWidth();
    window.addEventListener('resize', checkViewportWidth);
    return () => window.removeEventListener('resize', checkViewportWidth);
  }, []);

  if (!isScreenWideEnough) {
    return (
      <div
        style={{
          width: '100vw',
          height: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#050510',
          color: 'white',
          fontFamily: 'sans-serif',
          padding: '24px',
          textAlign: 'center',
        }}
      >
        <div>
          <h1 style={{ margin: '0 0 12px', fontSize: '28px' }}>Wider Screen Required</h1>
          <p style={{ margin: 0, opacity: 0.8, fontSize: '16px' }}>
            This experience requires a minimum width of {MIN_SCREEN_WIDTH}px.
          </p>
        </div>
      </div>
    );
  }

  return (
    // Main Wrapper
    <div style={{ width: '100vw', height: '100vh', position: 'relative', overflow: 'hidden' }}>
      
      {/* LAYER 1: The 3D Canvas (Background) */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
        <Scene />
      </div>

      {/* LAYER 2: The HTML UI (Foreground - "Glass Cockpit") */}
      <div 
        style={{ 
          position: 'absolute', 
          inset: 0, 
          zIndex: 1, 
          pointerEvents: 'none', // Lets mouse pass through to the 3D canvas...
          display: 'flex', 
          flexDirection: 'column', 
          justifyContent: 'space-between', 
          padding: '24px' 
        }}
      >
        {/* Top Header */}
        <header style={{ color: 'white', fontFamily: 'sans-serif' }}>
          <h1 style={{ margin: 0, fontSize: '24px', textShadow: '0 2px 10px rgba(0,0,0,0.5)' }}>
            AI Data Visualizer
          </h1>
          <p style={{ opacity: 0.7 }}>Rotate to explore the latent space.</p>
          <p style={{ opacity: 0.9, fontSize: '12px', marginTop: '6px' }}>Nodes in scene: {nodes.length}</p>
        </header>

        {/* Bottom Control Panel */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-end',
            gap: '16px',
            flexWrap: 'wrap',
          }}
        >
          <div
            style={{
              pointerEvents: 'auto',
              width: 'min(360px, 100%)',
              background: 'rgba(20, 20, 30, 0.8)',
              backdropFilter: 'blur(10px)',
              padding: '12px',
              borderRadius: '12px',
              border: '1px solid rgba(255,255,255,0.1)',
              overflow: 'hidden',
            }}
          >
            <FileUpload className="p-4" onChange={handleUploadFileSelection} />
            <div
              style={{
                marginTop: '8px',
                padding: '0 8px 8px',
                color: 'white',
                opacity: 0.8,
                fontSize: '12px',
                minHeight: '18px',
              }}
            >
              {isUploading ? `Uploading: ${uploadStatus}` : uploadStatus}
            </div>
            <div style={{ padding: '0 8px 8px' }}>
              <button
                type="button"
                onClick={handleConfirmUpload}
                disabled={isUploading || !pendingUploadFile}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  borderRadius: '6px',
                  border: 'none',
                  background: isUploading || !pendingUploadFile ? '#555' : '#22c55e',
                  color: isUploading || !pendingUploadFile ? '#aaa' : 'white',
                  fontWeight: 'bold',
                  cursor: isUploading || !pendingUploadFile ? 'not-allowed' : 'pointer',
                }}
              >
                {isUploading ? 'Uploading...' : 'Confirm Upload'}
              </button>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
            <div
              style={{
                pointerEvents: 'auto',
                background: 'rgba(20, 20, 30, 0.8)',
                backdropFilter: 'blur(10px)',
                padding: '16px',
                borderRadius: '12px',
                display: 'flex',
                gap: '12px',
                border: '1px solid rgba(255,255,255,0.1)'
              }}
            >
              <button
                type="button"
                disabled={isLoading || isUploading || !hasUploadedPdf}
                onClick={handleLoadGraph}
                style={{
                  padding: '10px 20px',
                  borderRadius: '6px',
                  border: 'none',
                  background: isLoading || isUploading || !hasUploadedPdf ? '#555' : 'cyan',
                  color: isLoading || isUploading || !hasUploadedPdf ? '#aaa' : 'black',
                  fontWeight: 'bold',
                  cursor: isLoading || isUploading || !hasUploadedPdf ? 'not-allowed' : 'pointer'
                }}
              >
                {isLoading ? 'Loading...' : 'Load Graph'}
              </button>
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (!queryText) return;
                const { nodes: currentNodes, edges: currentEdges } = useAppStore.getState();
                const query = queryText.toLowerCase();
                const matched = currentNodes.filter((n) =>
                  n.label.toLowerCase().includes(query) || n.text.toLowerCase().includes(query)
                );
                if (matched.length === 0) {
                  setActiveNodes([]);
                  return;
                }
                const matchedIds = new Set(matched.map((n) => n.id));
                for (const edge of currentEdges) {
                  if (matchedIds.has(edge.sourceId)) matchedIds.add(edge.targetId);
                  if (matchedIds.has(edge.targetId)) matchedIds.add(edge.sourceId);
                }
                setActiveNodes([...matchedIds]);
              }}
              style={{
                pointerEvents: 'auto',
                background: 'rgba(20, 20, 30, 0.8)',
                backdropFilter: 'blur(10px)',
                padding: '16px',
                borderRadius: '12px',
                display: 'flex',
                gap: '12px',
                border: '1px solid rgba(255,255,255,0.1)'
              }}
            >
              <input
                type="text"
                placeholder="Search nodes..."
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                style={{
                  width: '200px',
                  padding: '10px',
                  borderRadius: '6px',
                  border: 'none',
                  background: 'rgba(0,0,0,0.5)',
                  color: 'white'
                }}
              />
              <button
                type="submit"
                disabled={nodes.length === 0}
                style={{
                  padding: '10px 20px',
                  borderRadius: '6px',
                  border: 'none',
                  background: nodes.length === 0 ? '#555' : '#ff6b00',
                  color: nodes.length === 0 ? '#aaa' : 'white',
                  fontWeight: 'bold',
                  cursor: nodes.length === 0 ? 'not-allowed' : 'pointer'
                }}
              >
                Search
              </button>
            </form>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleAiQuery();
              }}
              style={{
                pointerEvents: 'auto',
                background: 'rgba(20, 20, 30, 0.8)',
                backdropFilter: 'blur(10px)',
                padding: '16px',
                borderRadius: '12px',
                display: 'flex',
                gap: '12px',
                border: '1px solid rgba(255,255,255,0.1)'
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <input
                  type="text"
                  placeholder="Ask a question..."
                  value={aiQuery}
                  onChange={(e) => setAiQuery(e.target.value)}
                  style={{
                    width: '250px',
                    padding: '10px',
                    borderRadius: '6px',
                    border: 'none',
                    background: 'rgba(0,0,0,0.5)',
                    color: 'white'
                  }}
                />
                <div style={{ color: 'white', fontSize: '12px', opacity: 0.8 }}>
                  Context size (k): {aiK}
                </div>
                <Slider
                  min={5}
                  max={20}
                  step={1}
                  value={[aiK]}
                  onValueChange={(values) => {
                    const next = values[0];
                    if (typeof next === 'number') setAiK(next);
                  }}
                />
              </div>
              <button
                type="submit"
                disabled={isQuerying || !aiQuery.trim()}
                style={{
                  padding: '10px 20px',
                  borderRadius: '6px',
                  border: 'none',
                  background: isQuerying || !aiQuery.trim() ? '#555' : '#a855f7',
                  color: isQuerying || !aiQuery.trim() ? '#aaa' : 'white',
                  fontWeight: 'bold',
                  cursor: isQuerying || !aiQuery.trim() ? 'not-allowed' : 'pointer'
                }}
              >
                {isQuerying ? 'Thinking...' : 'Ask AI'}
              </button>
            </form>
          </div>
        </div>
      </div>

      {/* LAYER 3: AI Answer Panel */}
      {aiAnswer !== null && (
        <div
          style={{
            position: 'absolute',
            top: '80px',
            left: '24px',
            width: '360px',
            maxHeight: 'calc(100vh - 160px)',
            zIndex: 2,
            pointerEvents: 'auto',
            background: 'rgba(20, 20, 30, 0.85)',
            backdropFilter: 'blur(10px)',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.1)',
            color: 'white',
            fontFamily: 'sans-serif',
            padding: '20px',
            overflowY: 'auto',
          }}
        >
          <button
            onClick={() => {
              setAiAnswer(null);
              setAiSources([]);
              setAiSourceNodes([]);
              setActiveNodes([]);
            }}
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              background: 'none',
              border: 'none',
              color: 'white',
              fontSize: '18px',
              cursor: 'pointer',
              opacity: 0.7,
            }}
          >
            ✕
          </button>

          <h3 style={{ margin: '0 0 12px', fontSize: '14px', opacity: 0.5, textTransform: 'uppercase', letterSpacing: '1px' }}>
            AI Answer
          </h3>
          <p style={{ margin: '0 0 16px', fontSize: '14px', lineHeight: 1.6, opacity: 0.9 }}>
            {aiAnswer}
          </p>

          {aiSources.length > 0 && (
            <>
              <h3 style={{ margin: '0 0 8px', fontSize: '14px', opacity: 0.5, textTransform: 'uppercase', letterSpacing: '1px' }}>
                Sources ({aiSources.length})
              </h3>
              {aiSources.map((src, i) => {
                const matchedNode = nodes.find((n) => n.text === src.text);
                return (
                  <div
                    key={i}
                    onClick={() => {
                      if (matchedNode) {
                        const clusterIds = new Set<string>([matchedNode.id]);
                        for (const edge of edges) {
                          if (edge.sourceId === matchedNode.id) clusterIds.add(edge.targetId);
                          if (edge.targetId === matchedNode.id) clusterIds.add(edge.sourceId);
                        }
                        const { activeNodeIds } = useAppStore.getState();
                        const mergedIds = new Set<string>([...activeNodeIds, ...clusterIds]);
                        setActiveNodes([...mergedIds]);
                        setSelectedNode(matchedNode.id);
                      }
                    }}
                    style={{
                      background: 'rgba(255,255,255,0.05)',
                      borderRadius: '8px',
                      padding: '10px 12px',
                      marginBottom: '8px',
                      fontSize: '12px',
                      cursor: matchedNode ? 'pointer' : 'default',
                      border: matchedNode ? '1px solid rgba(168,85,247,0.3)' : '1px solid transparent',
                      transition: 'border-color 0.2s',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', opacity: 0.5, marginBottom: '4px', fontSize: '11px' }}>
                      {matchedNode && (
                        <div style={{
                          width: '6px',
                          height: '6px',
                          borderRadius: '50%',
                          background: matchedNode.color,
                          flexShrink: 0,
                        }} />
                      )}
                      {src.metadata.source}
                      {matchedNode && <span style={{ opacity: 0.7 }}>— {matchedNode.label}</span>}
                    </div>
                    <div style={{ lineHeight: 1.5, opacity: 0.8 }}>
                      {src.text.length > 200 ? src.text.slice(0, 200) + '…' : src.text}
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      )}

      {/* LAYER 4: Node Detail Side Panel */}
      {(() => {
        const selectedNode = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;
        const isOpen = selectedNode !== null && selectedNode !== undefined;
        const neighbors = isOpen
          ? edges
              .filter((e) => e.sourceId === selectedNodeId || e.targetId === selectedNodeId)
              .map((e) => {
                const neighborId = e.sourceId === selectedNodeId ? e.targetId : e.sourceId;
                return nodes.find((n) => n.id === neighborId);
              })
              .filter(Boolean)
          : [];

        return (
          <div
            style={{
              position: 'absolute',
              top: 0,
              right: 0,
              bottom: 0,
              width: '320px',
              zIndex: 2,
              pointerEvents: isOpen ? 'auto' : 'none',
              transform: isOpen ? 'translateX(0)' : 'translateX(100%)',
              transition: 'transform 0.3s ease',
              background: 'rgba(20, 20, 30, 0.8)',
              backdropFilter: 'blur(10px)',
              borderLeft: '1px solid rgba(255,255,255,0.1)',
              color: 'white',
              fontFamily: 'sans-serif',
              padding: '24px',
              overflowY: 'auto',
            }}
          >
            {isOpen && (
              <>
                <button
                  onClick={() => {
                    setSelectedNode(null);
                    setActiveNodes([]);
                  }}
                  style={{
                    position: 'absolute',
                    top: '12px',
                    right: '12px',
                    background: 'none',
                    border: 'none',
                    color: 'white',
                    fontSize: '20px',
                    cursor: 'pointer',
                    opacity: 0.7,
                  }}
                >
                  ✕
                </button>

                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
                  <div
                    style={{
                      width: '12px',
                      height: '12px',
                      borderRadius: '50%',
                      background: selectedNode.color,
                      flexShrink: 0,
                    }}
                  />
                  <h2 style={{ margin: 0, fontSize: '20px' }}>{selectedNode.label}</h2>
                </div>

                <div style={{ marginBottom: '16px', opacity: 0.7, fontSize: '13px' }}>
                  <div>Color: {selectedNode.color}</div>
                  <div>
                    Position: ({selectedNode.position[0].toFixed(2)}, {selectedNode.position[1].toFixed(2)},{' '}
                    {selectedNode.position[2].toFixed(2)})
                  </div>
                </div>

                {selectedNode.text && (
                  <div style={{ marginBottom: '16px' }}>
                    <h3 style={{ margin: '0 0 8px', fontSize: '14px', opacity: 0.5, textTransform: 'uppercase', letterSpacing: '1px' }}>
                      Content
                    </h3>
                    <p style={{ margin: 0, fontSize: '13px', lineHeight: 1.5, opacity: 0.8 }}>
                      {selectedNode.text}
                    </p>
                  </div>
                )}

                <div>
                  <h3 style={{ margin: '0 0 8px', fontSize: '14px', opacity: 0.5, textTransform: 'uppercase', letterSpacing: '1px' }}>
                    Connections ({neighbors.length})
                  </h3>
                  {neighbors.length === 0 ? (
                    <p style={{ opacity: 0.4, fontSize: '13px' }}>No connections</p>
                  ) : (
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                      {neighbors.map((neighbor) => (
                        <li
                          key={neighbor!.id}
                          onClick={() => setSelectedNode(neighbor!.id)}
                          style={{
                            padding: '8px 10px',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            fontSize: '14px',
                            background: 'rgba(255,255,255,0.05)',
                            marginBottom: '4px',
                          }}
                        >
                          <div
                            style={{
                              width: '8px',
                              height: '8px',
                              borderRadius: '50%',
                              background: neighbor!.color,
                              flexShrink: 0,
                            }}
                          />
                          {neighbor!.label}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </>
            )}
          </div>
        );
      })()}
    </div>
  );
}