import express, { Request, Response, NextFunction } from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import {
  type CanonicalToolName,
  getAiStatus,
  handleAiToolRequest,
} from './api/_lib/aiEngine.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json({ limit: '10mb' }));

// Health / Status endpoint
app.get('/api/ai/status', (_req: Request, res: Response) => {
  res.setHeader('Cache-Control', 'no-store');
  res.json(getAiStatus());
});

// Register all 6 AI writing tool endpoints (supporting both canonical and alias paths)
function registerToolRoute(paths: string[], tool: CanonicalToolName) {
  app.post(paths, async (req: Request, res: Response) => {
    res.setHeader('Cache-Control', 'no-store');
    try {
      const outcome = await handleAiToolRequest(tool, req.body);
      return res.status(outcome.status).json(outcome.body);
    } catch (err: any) {
      console.error(`Unhandled error in /api/ai/${tool}:`, err);
      return res.status(500).json({
        success: false,
        error: 'An unexpected server error occurred. Please try again.',
      });
    }
  });
}

registerToolRoute(['/api/ai/humanize', '/api/ai/humanizer'], 'humanize');
registerToolRoute(['/api/ai/paraphrase', '/api/ai/paraphraser'], 'paraphrase');
registerToolRoute(['/api/ai/tone', '/api/ai/tone-changer'], 'tone');
registerToolRoute(['/api/ai/simplify', '/api/ai/simplifier'], 'simplify');
registerToolRoute(['/api/ai/summarize', '/api/ai/summarizer'], 'summarize');
registerToolRoute(['/api/ai/grammar', '/api/ai/grammar-fix'], 'grammar');

// Ensure unknown /api/* routes always return JSON (never HTML)
app.use('/api', (_req: Request, res: Response) => {
  res.status(404).json({ success: false, error: 'API endpoint not found.' });
});

// Express JSON error handler (catches malformed JSON bodies & unexpected errors)
app.use((err: any, req: Request, res: Response, next: NextFunction) => {
  if (req.path.startsWith('/api')) {
    const status = err?.status || 500;
    return res.status(status).json({
      success: false,
      error: err?.message || 'Internal server error',
    });
  }
  return next(err);
});

// Start Server with Vite Dev or Static Production
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Humanify AI Suite running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
