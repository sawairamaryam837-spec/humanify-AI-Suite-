export interface ApiResponse<T = any> {
  success?: boolean;
  result?: string;
  isDemoMode?: boolean;
  message?: string;
  error?: string;
  [key: string]: any;
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

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
        throw new Error('Empty response received from server. Please try again.');
      }

      let data: any;
      try {
        data = JSON.parse(responseText);
      } catch {
        if (res.status === 404) {
          throw new Error(
            `API endpoint ${endpointPath} was not found (404). Make sure backend API routes are deployed and GEMINI_API_KEY is configured.`
          );
        }
        throw new Error(`Unexpected response from server (status ${res.status}). Please try again.`);
      }

      if (!res.ok) {
        throw new Error(data.error || `Request failed with status ${res.status}.`);
      }

      return data;
    } catch (err: any) {
      const isAbort = err?.name === 'AbortError';
      const isRetryable =
        isAbort ||
        err?.message?.includes('Empty response') ||
        err?.message?.includes('Failed to fetch');

      if (attempt < maxAttempts && isRetryable) {
        await new Promise((r) => setTimeout(r, 400));
        continue;
      }

      const finalErr = isAbort
        ? new Error('The AI request took too long. Please try again.')
        : err;
      console.error(`AI API call to ${url} failed:`, finalErr);
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
