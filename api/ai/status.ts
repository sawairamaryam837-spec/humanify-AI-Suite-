import { getAiStatus } from '../_lib/aiEngine';
import type { ServerlessRequest, ServerlessResponse } from '../_lib/vercelAdapter';

export default function handler(req: ServerlessRequest, res: ServerlessResponse) {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');

  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.setHeader('Allow', 'GET, HEAD');
    return res.status(405).json({ success: false, error: 'Method not allowed. Use GET.' });
  }

  return res.status(200).json(getAiStatus());
}
