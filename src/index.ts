/**
 * Jarvis Memory - OpenClaw Plugin
 *
 * Long-term memory with pgvector and local embeddings (transformers.js)
 */

import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
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
} from './database.js';
import { embed, preloadModel } from './embeddings.js';

// Config
const MAX_MEMORIES_PER_USER = 1000;
const MIN_RESULTS_FOR_FALLBACK = 3;

const MEMORY_CATEGORIES = ["FACT", "PREFERENCE", "OPINION", "EVENT", "CONTEXT"] as const;
type MemoryCategory = typeof MEMORY_CATEGORIES[number];

// Plugin config schema
const configSchema = Type.Object({
  databaseUrl: Type.String({ description: "PostgreSQL connection URL with pgvector" }),
});

// Shared state
let dbInitialized = false;

async function ensureDbInitialized(dbUrl: string) {
  if (dbInitialized) return;

  initDatabase(dbUrl);
  await ensureSchema();
  await preloadModel();
  console.log('[jarvis-memory] Database and embeddings initialized');
  dbInitialized = true;
}

const jarvisMemoryPlugin = {
  id: "jarvis-memory",
  name: "Jarvis Memory",
  description: "Long-term memory with pgvector and local embeddings",
  kind: "memory" as const,
  configSchema,

  register(api: OpenClawPluginApi) {
    // Get config from plugin config
    const pluginConfig = api.pluginConfig as { databaseUrl?: string } | undefined;
    const dbUrl = pluginConfig?.databaseUrl || process.env.JARVIS_DATABASE_URL || null;

    if (!dbUrl) {
      console.error('[jarvis-memory] WARNING: Database URL not configured!');
      return;
    }

    console.log(`[jarvis-memory] Config: databaseUrl=SET`);

    // Register tool: memory_search
    api.registerTool(
      {
        name: "memory_search",
        label: "Memory Search",
        description: "Search through long-term memories using semantic similarity. Use when you need context about user preferences, past decisions, or previously discussed topics.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          user_id: Type.String({ description: "User ID (from Telegram, etc.)" }),
          limit: Type.Optional(Type.Number({ description: "Max results (default: 10)" })),
          threshold: Type.Optional(Type.Number({ description: "Similarity threshold 0-1 (default: 0.5)" })),
        }),
        async execute(_toolCallId: string, params: unknown) {
          const { query, user_id, limit = 10, threshold = 0.5 } = params as {
            query: string;
            user_id: string;
            limit?: number;
            threshold?: number;
          };

          try {
            await ensureDbInitialized(dbUrl);

            const queryEmbedding = await embed(query);
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

            if (results.length === 0) {
              return {
                content: [{ type: "text", text: "No relevant memories found." }],
                details: { count: 0 },
              };
            }

            const text = results
              .map((r, i) => `${i + 1}. [${r.memory_type}] ${r.content} (${(r.similarity * 100).toFixed(0)}%)`)
              .join("\n");

            const sanitizedResults = results.map((r) => ({
              id: r.id,
              content: r.content,
              type: r.memory_type,
              importance: r.importance,
              similarity: r.similarity,
            }));

            return {
              content: [{ type: "text", text: `Found ${results.length} memories:\n\n${text}` }],
              details: { count: results.length, memories: sanitizedResults },
            };
          } catch (error) {
            console.error('[jarvis-memory] Search error:', error);
            return {
              content: [{ type: "text", text: `Memory search error: ${error instanceof Error ? error.message : String(error)}` }],
              details: { error: true },
            };
          }
        },
      },
      { name: "memory_search" },
    );

    // Register tool: memory_store
    api.registerTool(
      {
        name: "memory_store",
        label: "Memory Store",
        description: "Save important information in long-term memory. Use for preferences, facts, decisions, events.",
        parameters: Type.Object({
          text: Type.String({ description: "Information to remember" }),
          user_id: Type.String({ description: "User ID (from Telegram, etc.)" }),
          importance: Type.Optional(Type.Number({ description: "Importance 0-1 (default: 0.7)" })),
          category: Type.Optional(
            Type.Unsafe<MemoryCategory>({
              type: "string",
              enum: [...MEMORY_CATEGORIES],
              description: "Memory category (default: FACT)",
            }),
          ),
        }),
        async execute(_toolCallId: string, params: unknown) {
          const {
            text,
            user_id,
            importance = 0.7,
            category = "FACT",
          } = params as {
            text: string;
            user_id: string;
            importance?: number;
            category?: MemoryCategory;
          };

          try {
            await ensureDbInitialized(dbUrl);

            // Check memory limit and evict if needed
            const currentCount = await countMemories(user_id);
            if (currentCount >= MAX_MEMORIES_PER_USER) {
              await evictLowest(user_id, 1);
            }

            // Generate embedding
            const embedding = await embed(text);

            // Check for duplicates
            const existing = await searchSimilar(user_id, embedding, 1, 0.95);
            if (existing.length > 0) {
              return {
                content: [{ type: "text", text: `Similar memory already exists: "${existing[0].content}"` }],
                details: { action: "duplicate", existingId: existing[0].id },
              };
            }

            // Store memory
            const id = await storeMemory(user_id, text, embedding, importance, category, {});

            console.log(`[jarvis-memory] Stored memory id=${id} for user=${user_id}`);

            return {
              content: [{ type: "text", text: `Stored: "${text.slice(0, 100)}${text.length > 100 ? '...' : ''}"` }],
              details: { action: "created", id },
            };
          } catch (error) {
            console.error('[jarvis-memory] Store error:', error);
            return {
              content: [{ type: "text", text: `Memory store error: ${error instanceof Error ? error.message : String(error)}` }],
              details: { error: true },
            };
          }
        },
      },
      { name: "memory_store" },
    );

    // Register tool: memory_forget
    api.registerTool(
      {
        name: "memory_forget",
        label: "Memory Forget",
        description: "Delete specific memories. GDPR-compliant.",
        parameters: Type.Object({
          user_id: Type.String({ description: "User ID" }),
          query: Type.Optional(Type.String({ description: "Search to find memory to delete" })),
          forget_all: Type.Optional(Type.Boolean({ description: "Delete ALL memories for user" })),
        }),
        async execute(_toolCallId: string, params: unknown) {
          const { user_id, query, forget_all } = params as {
            user_id: string;
            query?: string;
            forget_all?: boolean;
          };

          try {
            await ensureDbInitialized(dbUrl);

            if (forget_all) {
              const count = await forgetMemories(user_id);
              return {
                content: [{ type: "text", text: `Deleted all ${count} memories for user.` }],
                details: { action: "deleted_all", count },
              };
            }

            if (query) {
              const count = await forgetMemories(user_id, query);
              return {
                content: [{ type: "text", text: `Deleted ${count} memories matching "${query}"` }],
                details: { action: "deleted", count },
              };
            }

            return {
              content: [{ type: "text", text: "Provide query or forget_all=true." }],
              details: { error: "missing_param" },
            };
          } catch (error) {
            console.error('[jarvis-memory] Forget error:', error);
            return {
              content: [{ type: "text", text: `Memory forget error: ${error instanceof Error ? error.message : String(error)}` }],
              details: { error: true },
            };
          }
        },
      },
      { name: "memory_forget" },
    );

    // Register service for cleanup
    api.registerService({
      id: "jarvis-memory-cleanup",
      async start() {
        console.log("[jarvis-memory] Service started");
      },
      async stop() {
        await closeDatabase();
        dbInitialized = false;
        console.log("[jarvis-memory] Service stopped");
      },
    });

    console.log("[jarvis-memory] Plugin registered");
  },
};

export default jarvisMemoryPlugin;
