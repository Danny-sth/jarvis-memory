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
  getAllMemories,
  storeMemory,
  forgetMemories,
  countMemories,
  evictLowest,
  decayMemories,
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

// Session to user mapping (populated by message_received hook)
const sessionUserMap = new Map<string, { userId: string; channel: string }>();

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

        // Run decay on old memories at startup
        try {
          await ensureDbInitialized(dbUrl);
          await decayMemories(7, 0.95, 0.1); // 7 days, 5% decay, min 0.1
        } catch (error) {
          console.error('[jarvis-memory] Decay error:', error);
        }
      },
      async stop() {
        await closeDatabase();
        dbInitialized = false;
        sessionUserMap.clear();
        console.log("[jarvis-memory] Service stopped");
      },
    });

    // ===========================================
    // HOOK 1: message_received - Capture user_id
    // This is the PROPER way to get sender info
    // ===========================================
    api.on('message_received', async (
      event: { from?: string; content?: string; timestamp?: number; metadata?: Record<string, unknown> },
      ctx: Record<string, unknown>
    ) => {
      console.log(`[jarvis-memory] message_received: from=${event.from}, ctx=${JSON.stringify(ctx)}`);

      if (event.from) {
        const channel = String(ctx.channelId || 'unknown');
        // event.from may already have prefix (telegram:123) or be raw (123)
        const userId = event.from.includes(':') ? event.from : `${channel}:${event.from}`;

        // Store with multiple keys for reliable lookup
        const conversationId = String(ctx.conversationId || '');
        const sessionId = String(ctx.sessionId || '');

        // Primary key: conversationId (e.g., "telegram:764733417")
        if (conversationId) {
          sessionUserMap.set(conversationId, { userId, channel });
          console.log(`[jarvis-memory] Mapped conversationId=${conversationId} -> user=${userId}`);
        }

        // Fallback key: sessionId
        if (sessionId) {
          sessionUserMap.set(sessionId, { userId, channel });
        }

        // Extra fallback: channel:accountId
        if (ctx.channelId && ctx.accountId) {
          const altKey = `${ctx.channelId}:${ctx.accountId}`;
          sessionUserMap.set(altKey, { userId, channel });
        }
      }
    });

    // ===========================================
    // HOOK 2: before_prompt_build - Inject memory
    // Uses user_id captured from message_received
    // ===========================================
    api.on('before_prompt_build', async (
      event: { prompt: string; messages?: unknown[] },
      ctx: Record<string, unknown>
    ) => {
      try {
        console.log(`[jarvis-memory] PRE ctx=${JSON.stringify(ctx)}`);

        let userId: string | null = null;
        const sessionKey = String(ctx.sessionKey || '');
        const messageProvider = String(ctx.messageProvider || '');

        // Method 1: Extract from sessionKey pattern "agent:main:telegram:direct:USER_ID"
        if (sessionKey && messageProvider === 'telegram') {
          const match = sessionKey.match(/:direct:(\d+)$/);
          if (match) {
            userId = `telegram:${match[1]}`;
            console.log(`[jarvis-memory] PRE: Extracted userId from sessionKey: ${userId}`);
          }
        }

        // Method 2: Try sessionUserMap with various keys
        if (!userId) {
          const keysToTry = [
            ctx.conversationId,
            ctx.sessionId,
            // Try to construct conversationId from sessionKey
            sessionKey.includes(':direct:') ? `telegram:${sessionKey.split(':direct:')[1]}` : null,
            'default'
          ].filter(Boolean).map(String);

          for (const key of keysToTry) {
            if (sessionUserMap.has(key)) {
              const userInfo = sessionUserMap.get(key)!;
              userId = userInfo.userId;
              console.log(`[jarvis-memory] PRE: Got userId from map key=${key}: ${userId}`);
              break;
            }
          }
        }

        if (!userId) {
          console.log(`[jarvis-memory] PRE: No userId found, sessionKey=${sessionKey}`);
          return;
        }

        console.log(`[jarvis-memory] PRE: Loading ALL memories for user=${userId}`);

        await ensureDbInitialized(dbUrl);

        // Get ALL user memories - no limit, no filtering
        const results = await getAllMemories(userId);

        if (results.length === 0) {
          console.log('[jarvis-memory] PRE: No memories found');
          return;
        }

        // Format ALL memories as context
        const memoryText = results
          .map(m => `• [${m.memory_type}] ${m.content}`)
          .join('\n');

        console.log(`[jarvis-memory] PRE: Injecting ${results.length} memories`);

        // Return context to be prepended to user's message
        return {
          prependContext: `[Система: Релевантная информация из долгосрочной памяти]\n${memoryText}\n\n---\n\n`
        };
      } catch (error) {
        console.error('[jarvis-memory] PRE error:', error);
        return;
      }
    });

    // ===========================================
    // HOOK 3: agent_end - POST-processing
    // Auto-extract and store facts from conversation
    // ===========================================
    api.on('agent_end', async (
      event: { messages?: unknown[]; success?: boolean; error?: string },
      ctx: Record<string, unknown>
    ) => {
      try {
        if (!event.success || !event.messages || event.messages.length < 2) {
          return;
        }

        // Extract userId from sessionKey
        const sessionKey = String(ctx.sessionKey || '');
        const messageProvider = String(ctx.messageProvider || '');

        let userId: string | null = null;
        if (sessionKey && messageProvider === 'telegram') {
          const match = sessionKey.match(/:direct:(\d+)$/);
          if (match) {
            userId = `telegram:${match[1]}`;
          }
        }

        if (!userId) {
          return; // Can't store without user_id
        }

        // Get last user message and assistant response
        const messages = event.messages as Array<{ role?: string; content?: unknown }>;
        const lastUserMsg = messages.filter(m => m.role === 'user').pop();
        const lastAssistantMsg = messages.filter(m => m.role === 'assistant').pop();

        if (!lastUserMsg?.content || !lastAssistantMsg?.content) {
          return;
        }

        // Extract text content (may be string or array of content blocks)
        const extractText = (content: unknown): string => {
          if (typeof content === 'string') return content;
          if (Array.isArray(content)) {
            return content
              .filter((c: { type?: string }) => c.type === 'text')
              .map((c: { text?: string }) => c.text || '')
              .join(' ');
          }
          return '';
        };

        const userText = extractText(lastUserMsg.content);
        if (!userText) {
          return;
        }
        const extractedFacts: Array<{ text: string; category: string }> = [];

        // Check for name
        const nameMatch = userText.match(/меня зовут\s+([А-Яа-яA-Za-z]+)/i);
        if (nameMatch) {
          extractedFacts.push({ text: `Имя - ${nameMatch[1]}`, category: 'FACT' });
        }

        // Check for age
        const ageMatch = userText.match(/мне\s+(\d+)\s*(?:лет|год)/i);
        if (ageMatch) {
          extractedFacts.push({ text: `Возраст - ${ageMatch[1]} лет`, category: 'FACT' });
        }

        // Check for location
        const locationMatch = userText.match(/(?:живу в|из города?)\s+([А-Яа-яA-Za-z\s]+?)(?:\.|,|$)/i);
        if (locationMatch) {
          extractedFacts.push({ text: `Живет в ${locationMatch[1].trim()}`, category: 'FACT' });
        }

        // Check for preferences
        const prefMatch = userText.match(/(?:люблю|нравится|обожаю)\s+(.+?)(?:\.|,|!|$)/i);
        if (prefMatch && prefMatch[1].length < 100) {
          extractedFacts.push({ text: `Любит ${prefMatch[1].trim()}`, category: 'PREFERENCE' });
        }

        if (extractedFacts.length === 0) {
          return;
        }

        console.log(`[jarvis-memory] POST: Found ${extractedFacts.length} facts for user=${userId}`);

        await ensureDbInitialized(dbUrl);

        for (const fact of extractedFacts) {
          const embedding = await embed(fact.text);

          // Check for duplicates
          const existing = await searchSimilar(userId, embedding, 1, 0.9);
          if (existing.length > 0) {
            console.log(`[jarvis-memory] POST: Skip duplicate "${fact.text}"`);
            continue;
          }

          await storeMemory(userId, fact.text, embedding, 0.8, fact.category, { source: 'auto_extract' });
          console.log(`[jarvis-memory] POST: Stored "${fact.text}"`);
        }
      } catch (error) {
        console.error('[jarvis-memory] POST error:', error);
      }
    });

    console.log("[jarvis-memory] Plugin registered with PRE + POST hooks");
  },
};

export default jarvisMemoryPlugin;
