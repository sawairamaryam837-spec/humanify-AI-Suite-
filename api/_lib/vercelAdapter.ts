import type { IncomingMessage } from 'http';
import { CanonicalToolName, handleAiToolRequest } from './aiEngine';

export interface ServerlessRequest extends IncomingMessage {
  method?: string;
  body?: any;
  query?: Record<string, string | string[]>;
}

export interface ServerlessResponse {
  status: (code: number) => ServerlessResponse;
  json: (data: any) => void;
  setHeader: (name: string, value: string) => void;
  end: (chunk?: any) => void;
}

async function parseJsonBody(req: ServerlessRequest): Promise<Record<string, any> | null> {
  if (req.body !== undefined && req.body !== null) {
    if (typeof req.body === 'string') {
      try {
        return JSON.parse(req.body);
      } catch {
        return null;
      }
    }
    if (typeof req.body === 'object') {
      return req.body;
    }
  }

  // Fallback for raw Node streams when body parser wasn't pre-applied
  return new Promise((resolve) => {
    let raw = '';
    req.on('data', (chunk) => {
      raw += chunk;
      if (raw.length > 10 * 1024 * 1024) {
        req.destroy();
        resolve(null);
      }
    });
    req.on('end', () => {
      if (!raw.trim()) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch {
        resolve(null);
      }
    });
    req.on('error', () => resolve(null));
  });
}

export function createToolHandler(tool: CanonicalToolName) {
  return async function handler(req: ServerlessRequest, res: ServerlessResponse) {
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Cache-Control', 'no-store');

    if (req.method === 'OPTIONS') {
      res.setHeader('Allow', 'POST, OPTIONS');
      return res.status(204).end();
    }

    if (req.method !== 'POST') {
      res.setHeader('Allow', 'POST, OPTIONS');
      return res.status(405).json({
        success: false,
        error: `Method ${req.method || 'UNKNOWN'} not allowed. Use POST.`,
      });
    }

    const body = await parseJsonBody(req);
    if (!body) {
      return res.status(400).json({
        success: false,
        error: 'Invalid JSON request body.',
      });
    }

    try {
      const outcome = await handleAiToolRequest(tool, body);
      return res.status(outcome.status).json(outcome.body);
    } catch (err: any) {
      console.error(`Unhandled error in /api/ai/${tool}:`, err);
      return res.status(500).json({
        success: false,
        error: 'An unexpected server error occurred. Please try again.',
      });
    }
  };
}
