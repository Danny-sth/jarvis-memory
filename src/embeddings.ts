/**
 * Embeddings module using transformers.js (ONNX)
 * Uses paraphrase-multilingual-MiniLM-L12-v2 for Russian support
 */

import { pipeline, type FeatureExtractionPipeline } from '@xenova/transformers';

const MODEL_NAME = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';
const DIMENSIONS = 384;

let embedder: FeatureExtractionPipeline | null = null;
let loadingPromise: Promise<FeatureExtractionPipeline> | null = null;

/**
 * Get or initialize the embedding model (singleton)
 */
async function getEmbedder(): Promise<FeatureExtractionPipeline> {
  if (embedder) return embedder;

  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    console.log(`[jarvis-memory] Loading embedding model: ${MODEL_NAME}`);
    const model = await pipeline('feature-extraction', MODEL_NAME);
    console.log(`[jarvis-memory] Embedding model loaded`);
    embedder = model;
    return model;
  })();

  return loadingPromise;
}

/**
 * Generate embedding for text
 */
export async function embed(text: string): Promise<number[]> {
  const model = await getEmbedder();

  const output = await model(text, {
    pooling: 'mean',
    normalize: true
  });

  // Convert to array
  const embedding = Array.from(output.data as Float32Array);

  if (embedding.length !== DIMENSIONS) {
    throw new Error(`Expected ${DIMENSIONS} dimensions, got ${embedding.length}`);
  }

  return embedding;
}

/**
 * Convert embedding array to PostgreSQL vector string
 */
export function toVectorString(embedding: number[]): string {
  return '[' + embedding.join(',') + ']';
}

/**
 * Preload the model (call at plugin init)
 */
export async function preloadModel(): Promise<void> {
  await getEmbedder();
}

export { DIMENSIONS };
