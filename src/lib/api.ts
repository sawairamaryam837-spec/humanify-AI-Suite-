export interface ApiResponse<T = any> {
  success?: boolean;
  result?: string;
  isDemoMode?: boolean;
  message?: string;
  error?: string | { message?: string; code?: string | number };
  code?: string | number;
  [key: string]: any;
}

const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL || '').replace(/\/+$/, '');

const toolEndpoints: Record<string, string> = {
  humanizer: '/api/ai/humanize',
  paraphraser: '/api/ai/paraphrase',
  'tone-changer': '/api/ai/tone',
  simplifier: '/api/ai/simplify',
  summarizer: '/api/ai/summarize',
  'grammar-fix': '/api/ai/grammar',
};

function buildApiUrl(path: string): string {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

/**
 * Safely extracts a human-readable string from any thrown error or API JSON payload.
 * Guarantees "[object Object]" is never surfaced to the user.
 */
export function extractErrorMessage(input: unknown, fallback = 'An unexpected AI error occurred. Please try again.'): string {
  if (typeof input === 'string') {
    const trimmed = input.trim();
    if (trimmed && trimmed !== '[object Object]') {
      return trimmed;
    }
    return fallback;
  }

  if (input instanceof Error) {
    const msg = input.message?.trim();
    if (msg && msg !== '[object Object]') {
      return msg;
    }
    const cause = (input as any).cause;
    if (cause) {
      return extractErrorMessage(cause, fallback);
    }
    return fallback;
  }

  if (input && typeof input === 'object') {
    const obj = input as Record<string, any>;

    // Handle nested { error: { message, code } } (e.g., Vercel or Gemini API error envelope)
    if (obj.error !== undefined && obj.error !== null) {
      const nested = extractErrorMessage(obj.error, '');
      if (nested) {
        const code = typeof obj.error === 'object' ? obj.error.code : obj.code;
        if (code && !nested.includes(String(code))) {
          return `${nested} (${code})`;
        }
        return nested;
      }
    }

    if (typeof obj.message === 'string' && obj.message.trim() && obj.message.trim() !== '[object Object]') {
      return obj.message.trim();
    }

    if (typeof obj.detail === 'string' && obj.detail.trim()) {
      return obj.detail.trim();
    }

    if (typeof obj.details === 'string' && obj.details.trim()) {
      return obj.details.trim();
    }

    if (typeof obj.statusText === 'string' && obj.statusText.trim()) {
      return obj.statusText.trim();
    }

    if (obj.code !== undefined) {
      return `Server returned error code ${String(obj.code)}. Please try again.`;
    }
  }

  return fallback;
}

function formatHttpStatusError(status: number, endpointPath: string, data?: any): string {
  const extracted = data ? extractErrorMessage(data, '') : '';

  if (status === 401 || status === 403) {
    return extracted || 'API authentication failed. Please verify your GEMINI_API_KEY in server environment settings.';
  }
  if (status === 404) {
    return extracted || `API endpoint ${endpointPath} was not found (404). Make sure backend API routes are deployed.`;
  }
  if (status === 429) {
    return extracted || 'AI rate limit reached temporarily. Please wait a few seconds and try again.';
  }
  if (status === 503) {
    return extracted || 'The AI service is temporarily busy or missing GEMINI_API_KEY configuration. Please try again shortly.';
  }
  if (status === 504) {
    return extracted || 'The AI server timed out before responding. Please try again with shorter text or retry in a moment.';
  }
  if (status >= 500) {
    return extracted
      ? `Server error (${status}): ${extracted}`
      : `A server error (${status}) occurred while processing your request. Please check your deployment logs and GEMINI_API_KEY.`;
  }
  return extracted || `Request failed with status ${status}. Please try again.`;
}

export async function processAiTool(toolId: string, payload: Record<string, any>): Promise<ApiResponse> {
  const endpointPath = toolEndpoints[toolId] || `/api/ai/${toolId}`;
  const url = buildApiUrl(endpointPath);
  const maxAttempts = 2;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const responseText = await res.text();
      if (!responseText || !responseText.trim()) {
        throw new Error(
          res.ok
            ? 'Empty response received from server. Please try again.'
            : formatHttpStatusError(res.status, endpointPath)
        );
      }

      let data: any;
      try {
        data = JSON.parse(responseText);
      } catch {
        throw new Error(formatHttpStatusError(res.status, endpointPath));
      }

      if (!res.ok) {
        throw new Error(formatHttpStatusError(res.status, endpointPath, data));
      }

      if (data?.success === false) {
        throw new Error(extractErrorMessage(data, 'AI processing failed. Please try again.'));
      }

      return data;
    } catch (err: any) {
      const isAbort = err?.name === 'AbortError';
      const rawMsg = extractErrorMessage(err, '');
      const isNetworkFailure =
        err instanceof TypeError ||
        rawMsg.includes('Failed to fetch') ||
        rawMsg.includes('NetworkError') ||
        rawMsg.includes('Load failed');
      const isRetryable =
        isAbort ||
        isNetworkFailure ||
        rawMsg.includes('Empty response');

      if (attempt < maxAttempts && isRetryable) {
        await new Promise((r) => setTimeout(r, 400));
        continue;
      }

      const finalMessage = isAbort
        ? 'The AI request took too long and timed out. Please try again.'
        : isNetworkFailure
        ? 'Network error: Unable to reach the AI backend. Please check your internet connection and try again.'
        : extractErrorMessage(err, 'Unable to complete AI request. Please try again.');

      const finalErr = new Error(finalMessage);
      console.error(`AI API call to ${url} failed:`, finalMessage, err);
      throw finalErr;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  throw new Error('Unable to complete AI request. Please try again.');
}

export async function checkAiHealth(): Promise<{ hasKey: boolean; model: string; status: string }> {
  try {
    const res = await fetch(buildApiUrl('/api/ai/status'), {
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) throw new Error('Status failed');
    return await res.json();
  } catch {
    return { hasKey: false, model: 'unavailable', status: 'offline' };
  }
}
