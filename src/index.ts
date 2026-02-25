/**
 * Jarvis Memory - OpenClaw Plugin
 *
 * Long-term memory with pgvector and local embeddings (transformers.js)
 */

import {
  initDatabase,
  ensureSchema,
  searchSimilar,
  searchText,
  storeMemory,
  forgetMemories,
  countMemories,
  evictLowest,
  closeDatabase,
  type SearchResult
} from './database.js';

import { embed, preloadModel } from './embeddings.js';

// Config
const MAX_MEMORIES_PER_USER = 1000;
const MIN_RESULTS_FOR_FALLBACK = 3;

interface PluginConfig {
  databaseUrl: string;
}

interface ToolContext {
  userId?: string;
  channelId?: string;
}

/**
 * OpenClaw Plugin Registration
 */
export default function register(api: any) {
  const config: PluginConfig = api.config || {};
  let initialized = false;

  // Initialize on first use
  async function ensureInitialized() {
    if (initialized) return;

    const dbUrl = config.databaseUrl || process.env.JARVIS_DATABASE_URL;
    if (!dbUrl) {
      throw new Error('Database URL not configured. Set JARVIS_DATABASE_URL or plugin config.databaseUrl');
    }

    initDatabase(dbUrl);
    await ensureSchema();
    await preloadModel();
    console.log('[jarvis-memory] Plugin initialized');
    initialized = true;
  }

  // Register tool: memory_search
  api.registerTool?.({
    name: 'memory_search',
    description: 'Search through long-term memories using semantic similarity. Use this to recall past conversations, facts about the user, preferences, etc.',
    parameters: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'The search query - what to look for in memories'
        },
        user_id: {
          type: 'string',
          description: 'The user ID (typically from Telegram)'
        },
        limit: {
          type: 'number',
          description: 'Maximum number of results (default: 10)'
        },
        threshold: {
          type: 'number',
          description: 'Similarity threshold 0-1 (default: 0.7)'
        }
      },
      required: ['query', 'user_id']
    },
    handler: async (params: { query: string; user_id: string; limit?: number; threshold?: number }, ctx: ToolContext) => {
      try {
        await ensureInitialized();
        const { query, user_id, limit = 10, threshold = 0.7 } = params;

        // Generate embedding for query
        const queryEmbedding = await embed(query);

        // Search similar memories
        let results = await searchSimilar(user_id, queryEmbedding, limit, threshold);

        // Fallback to text search for proper nouns if few results
        if (results.length < MIN_RESULTS_FOR_FALLBACK && query.split(' ').length <= 3) {
          const textResults = await searchText(user_id, query, limit);
          const seenIds = new Set(results.map(r => r.id));

          for (const tr of textResults) {
            if (!seenIds.has(tr.id)) {
              results.push({ ...tr, similarity: 0.5 });
              seenIds.add(tr.id);
            }
          }
        }

        // Format results
        const memories = results.map(r => ({
          id: r.id,
          content: r.content,
          importance: r.importance,
          type: r.memory_type,
          similarity: r.similarity,
          created_at: r.created_at
        }));

        return {
          success: true,
          data: {
            query,
            count: memories.length,
            memories
          }
        };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : String(error)
        };
      }
    }
  });

  // Register tool: memory_store
  api.registerTool?.({
    name: 'memory_store',
    description: 'Store a new memory about the user. Use this to remember important facts, preferences, events, or any information worth keeping.',
    parameters: {
      type: 'object',
      properties: {
        content: {
          type: 'string',
          description: 'The memory content to store'
        },
        user_id: {
          type: 'string',
          description: 'The user ID (typically from Telegram)'
        },
        importance: {
          type: 'number',
          description: 'Importance score 0-1 (default: 0.5). Higher = less likely to be forgotten'
        },
        memory_type: {
          type: 'string',
          enum: ['FACT', 'PREFERENCE', 'OPINION', 'EVENT', 'CONTEXT'],
          description: 'Type of memory (default: FACT)'
        },
        metadata: {
          type: 'object',
          description: 'Additional metadata to store with the memory'
        }
      },
      required: ['content', 'user_id']
    },
    handler: async (params: { content: string; user_id: string; importance?: number; memory_type?: string; metadata?: Record<string, unknown> }, ctx: ToolContext) => {
      try {
        await ensureInitialized();
        const {
          content,
          user_id,
          importance = 0.5,
          memory_type = 'FACT',
          metadata = {}
        } = params;

        // Check memory limit
        const currentCount = await countMemories(user_id);
        if (currentCount >= MAX_MEMORIES_PER_USER) {
          await evictLowest(user_id, 1);
        }

        // Generate embedding
        const embedding = await embed(content);

        // Store memory
        const id = await storeMemory(
          user_id,
          content,
          embedding,
          importance,
          memory_type,
          metadata
        );

        return {
          success: true,
          data: {
            id,
            message: 'Memory stored successfully'
          }
        };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : String(error)
        };
      }
    }
  });

  // Register tool: memory_forget
  api.registerTool?.({
    name: 'memory_forget',
    description: 'Delete memories. Can delete specific memories matching a query or all memories for a user.',
    parameters: {
      type: 'object',
      properties: {
        user_id: {
          type: 'string',
          description: 'The user ID'
        },
        query: {
          type: 'string',
          description: 'Delete memories containing this text'
        },
        forget_all: {
          type: 'boolean',
          description: 'If true, delete ALL memories for this user'
        }
      },
      required: ['user_id']
    },
    handler: async (params: { user_id: string; query?: string; forget_all?: boolean }, ctx: ToolContext) => {
      try {
        await ensureInitialized();
        const { user_id, query, forget_all = false } = params;

        if (forget_all) {
          const count = await forgetMemories(user_id);
          return {
            success: true,
            data: {
              deleted: count,
              message: `Deleted all ${count} memories`
            }
          };
        }

        if (!query) {
          return {
            success: false,
            error: 'Either query or forget_all must be provided'
          };
        }

        const count = await forgetMemories(user_id, query);

        return {
          success: true,
          data: {
            deleted: count,
            message: `Deleted ${count} memories matching "${query}"`
          }
        };
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : String(error)
        };
      }
    }
  });

  // Register cleanup service
  api.registerService?.({
    id: 'jarvis-memory-cleanup',
    start: async () => {
      console.log('[jarvis-memory] Service started');
    },
    stop: async () => {
      await closeDatabase();
      console.log('[jarvis-memory] Service stopped');
    }
  });

  console.log('[jarvis-memory] Plugin registered');
}
