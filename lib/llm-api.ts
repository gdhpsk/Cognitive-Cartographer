type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null;
}

async function readJsonSafe(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function extractDetail(payload: unknown): string {
  if (!isRecord(payload)) return '';
  return typeof payload.detail === 'string' ? payload.detail : '';
}

function toNonNegativeInt(value: unknown, fallback: number): number {
  const parsed =
    typeof value === 'number'
      ? value
      : typeof value === 'string'
        ? Number(value)
        : NaN;

  if (!Number.isFinite(parsed) || parsed < 0) return fallback;
  return Math.floor(parsed);
}

export interface ModelPolicy {
  availableModels: string[];
  allowCustomHfModels: boolean;
  maxPdfSizeMb: number;
  maxSeqLen: number;
}

export async function fetchModelPolicy(apiBase: string): Promise<ModelPolicy> {
  const response = await fetch(`${apiBase}/aval_model`, { cache: 'no-store' });
  const payload = await readJsonSafe(response);

  if (!response.ok) {
    const detail = extractDetail(payload);
    throw new Error(detail || `Failed to load model policy (HTTP ${response.status}).`);
  }

  const availableModels =
    isRecord(payload) && Array.isArray(payload.available_models)
      ? payload.available_models
          .filter((model): model is string => typeof model === 'string')
          .map((model) => model.trim())
          .filter((model) => model.length > 0)
      : [];

  const allowCustomHfModels =
    isRecord(payload) && typeof payload.allow_custom_hf_models === 'boolean'
      ? payload.allow_custom_hf_models
      : false;

  const maxPdfSizeMb =
    isRecord(payload)
      ? toNonNegativeInt(payload.max_pdf_size_mb, 0)
      : 0;

  const maxSeqLen =
    isRecord(payload)
      ? toNonNegativeInt(payload.max_seq_len, 0)
      : 0;

  return {
    availableModels: [...new Set(availableModels)],
    allowCustomHfModels,
    maxPdfSizeMb,
    maxSeqLen,
  };
}

export interface ChooseLlmResult {
  ok: boolean;
  statusCode: number;
  detail: string;
}

export async function chooseLlmModel(apiBase: string, modelName: string): Promise<ChooseLlmResult> {
  const response = await fetch(`${apiBase}/choose_llm`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_name: modelName,
    }),
  });

  const payload = await readJsonSafe(response);
  const detail = extractDetail(payload);
  const status =
    isRecord(payload) && typeof payload.status === 'string' ? payload.status : '';

  return {
    ok: response.ok && (status === '' || status === 'ok'),
    statusCode: response.status,
    detail,
  };
}
