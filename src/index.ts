/**
 * Jarvis Memory - OpenClaw Plugin
 * Long-term memory with pgvector, local embeddings, and LLM-based fact extraction
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

const MAX_MEMORIES_PER_USER = 1000;
const MEMORY_CATEGORIES = ["FACT", "PREFERENCE", "OPINION", "EVENT", "CONTEXT"] as const;

// Logger with levels
const LOG_LEVEL = process.env.JARVIS_MEMORY_LOG_LEVEL || 'info';
const LEVELS = { error: 0, warn: 1, info: 2, debug: 3 } as const;
const currentLevel = LEVELS[LOG_LEVEL as keyof typeof LEVELS] ?? LEVELS.info;

const log = {
  error: (...args: unknown[]) => currentLevel >= LEVELS.error && console.error('[jarvis-memory]', ...args),
  warn: (...args: unknown[]) => currentLevel >= LEVELS.warn && console.warn('[jarvis-memory]', ...args),
  info: (...args: unknown[]) => currentLevel >= LEVELS.info && console.log('[jarvis-memory]', ...args),
  debug: (...args: unknown[]) => currentLevel >= LEVELS.debug && console.log('[jarvis-memory] [DEBUG]', ...args),
};
type MemoryCategory = typeof MEMORY_CATEGORIES[number];

const configSchema = Type.Object({
  databaseUrl: Type.String({ description: "PostgreSQL connection URL with pgvector" }),
  anthropicApiKey: Type.Optional(Type.String({ description: "Anthropic API key for LLM fact extraction" })),
  extractionModel: Type.Optional(Type.String({ description: "Model for fact extraction (default: claude-haiku-4-5-20251001)" })),
});

let dbInitialized = false;

async function ensureDbInitialized(dbUrl: string) {
  if (dbInitialized) return;
  initDatabase(dbUrl);
  await ensureSchema();
  await preloadModel();
  log.info('Database and embedding model initialized');
  dbInitialized = true;
}

function extractUserId(ctx: Record<string, unknown>): string | null {
  const sessionKey = String(ctx.sessionKey || '');
  const messageProvider = String(ctx.messageProvider || '');

  if (sessionKey && messageProvider === 'telegram') {
    const match = sessionKey.match(/:direct:(\d+)$/);
    if (match) return `telegram:${match[1]}`;
  }
  return null;
}

function extractTextContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter((c: { type?: string }) => c.type === 'text')
      .map((c: { text?: string }) => c.text || '')
      .join(' ');
  }
  return '';
}

function stripMetadata(text: string): string {
  const match = text.match(/```\s*\n\n([^`]+)$/);
  return match ? match[1].trim() : text;
}

const jarvisMemoryPlugin = {
  id: "jarvis-memory",
  name: "Jarvis Memory",
  description: "Long-term memory with pgvector and local embeddings",
  kind: "memory" as const,
  configSchema,

  register(api: OpenClawPluginApi) {
    const pluginConfig = api.pluginConfig as {
      databaseUrl?: string;
      anthropicApiKey?: string;
      extractionModel?: string;
    } | undefined;

    const dbUrl = pluginConfig?.databaseUrl || process.env.JARVIS_DATABASE_URL || null;
    const anthropicApiKey = pluginConfig?.anthropicApiKey || process.env.ANTHROPIC_API_KEY || null;
    const extractionModel = pluginConfig?.extractionModel || 'claude-haiku-4-5-20251001';

    if (!dbUrl) {
      log.error('No database URL configured');
      return;
    }

    if (!anthropicApiKey) {
      log.warn('No Anthropic API key - automatic fact extraction disabled');
    }

    log.debug('Config:', { dbUrl: dbUrl.replace(/:[^:@]+@/, ':***@'), model: extractionModel, hasApiKey: !!anthropicApiKey });

    // Tool: memory_search
    api.registerTool({
      name: "memory_search",
      label: "Memory Search",
      description: "Search memories using semantic similarity",
      parameters: Type.Object({
        query: Type.String({ description: "Search query" }),
        user_id: Type.String({ description: "User ID" }),
        limit: Type.Optional(Type.Number({ description: "Max results (default: 10)" })),
        threshold: Type.Optional(Type.Number({ description: "Similarity threshold 0-1 (default: 0.5)" })),
      }),
      async execute(_toolCallId: string, params: unknown) {
        const { query, user_id, limit = 10, threshold = 0.5 } = params as {
          query: string; user_id: string; limit?: number; threshold?: number;
        };

        try {
          await ensureDbInitialized(dbUrl);
          const queryEmbedding = await embed(query);
          let results = await searchSimilar(user_id, queryEmbedding, limit, threshold);

          // Fallback to text search if few results
          if (results.length < 3 && query.split(' ').length <= 3) {
            const textResults = await searchText(user_id, query, limit);
            const seenIds = new Set(results.map(r => r.id));
            for (const tr of textResults) {
              if (!seenIds.has(tr.id)) {
                results.push({ ...tr, similarity: 0.5 });
              }
            }
          }

          if (results.length === 0) {
            return { content: [{ type: "text", text: "No memories found." }], details: { count: 0 } };
          }

          const text = results.map((r, i) =>
            `${i + 1}. [${r.memory_type}] ${r.content} (${(r.similarity * 100).toFixed(0)}%)`
          ).join("\n");

          return {
            content: [{ type: "text", text: `Found ${results.length} memories:\n\n${text}` }],
            details: { count: results.length },
          };
        } catch (error) {
          log.error('Search error:', error);
          return { content: [{ type: "text", text: `Error: ${error}` }], details: { error: true } };
        }
      },
    }, { name: "memory_search" });

    // Tool: memory_store
    api.registerTool({
      name: "memory_store",
      label: "Memory Store",
      description: "Save information in long-term memory",
      parameters: Type.Object({
        text: Type.String({ description: "Information to remember" }),
        user_id: Type.String({ description: "User ID" }),
        importance: Type.Optional(Type.Number({ description: "Importance 0-1 (default: 0.7)" })),
        category: Type.Optional(Type.Unsafe<MemoryCategory>({
          type: "string", enum: [...MEMORY_CATEGORIES], description: "Category (default: FACT)"
        })),
      }),
      async execute(_toolCallId: string, params: unknown) {
        const { text, user_id, importance = 0.7, category = "FACT" } = params as {
          text: string; user_id: string; importance?: number; category?: MemoryCategory;
        };

        try {
          await ensureDbInitialized(dbUrl);

          const currentCount = await countMemories(user_id);
          if (currentCount >= MAX_MEMORIES_PER_USER) {
            await evictLowest(user_id, 1);
          }

          const embedding = await embed(text);
          const existing = await searchSimilar(user_id, embedding, 1, 0.95);
          if (existing.length > 0) {
            return { content: [{ type: "text", text: `Duplicate: "${existing[0].content}"` }], details: { action: "duplicate" } };
          }

          const id = await storeMemory(user_id, text, embedding, importance, category, {});
          log.info(`Stored memory id=${id} for user=${user_id}, category=${category}`);
          log.debug('Stored content:', text.slice(0, 100));

          return { content: [{ type: "text", text: `Stored: "${text.slice(0, 100)}"` }], details: { action: "created", id } };
        } catch (error) {
          log.error('Store error:', error);
          return { content: [{ type: "text", text: `Error: ${error}` }], details: { error: true } };
        }
      },
    }, { name: "memory_store" });

    // Tool: memory_forget
    api.registerTool({
      name: "memory_forget",
      label: "Memory Forget",
      description: "Delete memories",
      parameters: Type.Object({
        user_id: Type.String({ description: "User ID" }),
        query: Type.Optional(Type.String({ description: "Search query to find memories to delete" })),
        forget_all: Type.Optional(Type.Boolean({ description: "Delete ALL memories" })),
      }),
      async execute(_toolCallId: string, params: unknown) {
        const { user_id, query, forget_all } = params as {
          user_id: string; query?: string; forget_all?: boolean;
        };

        try {
          await ensureDbInitialized(dbUrl);

          if (forget_all) {
            const count = await forgetMemories(user_id);
            return { content: [{ type: "text", text: `Deleted ${count} memories` }], details: { count } };
          }
          if (query) {
            const count = await forgetMemories(user_id, query);
            return { content: [{ type: "text", text: `Deleted ${count} memories matching "${query}"` }], details: { count } };
          }
          return { content: [{ type: "text", text: "Provide query or forget_all=true" }], details: { error: "missing_param" } };
        } catch (error) {
          log.error('Forget error:', error);
          return { content: [{ type: "text", text: `Error: ${error}` }], details: { error: true } };
        }
      },
    }, { name: "memory_forget" });

    // Service
    api.registerService({
      id: "jarvis-memory-service",
      async start() {
        try {
          await ensureDbInitialized(dbUrl);
          const decayed = await decayMemories(7, 0.95, 0.1);
          log.info(`Service started, decayed ${decayed} old memories`);
        } catch (error) {
          log.error('Startup error:', error);
        }
      },
      async stop() {
        await closeDatabase();
        dbInitialized = false;
      },
    });

    // Hook: PRE - Inject memories into context
    api.on('before_prompt_build', async (
      _event: { prompt: string; messages?: unknown[] },
      ctx: Record<string, unknown>
    ) => {
      try {
        const userId = extractUserId(ctx);
        if (!userId) {
          log.debug('PRE: No userId extracted from context');
          return;
        }

        await ensureDbInitialized(dbUrl);
        const results = await getAllMemories(userId);

        if (results.length === 0) {
          log.debug(`PRE: No memories for user=${userId}`);
          return;
        }

        const memoryText = results.map(m => `• [${m.memory_type}] ${m.content}`).join('\n');
        log.info(`PRE: Injecting ${results.length} memories for user=${userId}`);
        log.debug('PRE: Memory types:', results.map(m => m.memory_type));

        return {
          prependContext: `[Память пользователя]\n${memoryText}\n\n---\n\n`
        };
      } catch (error) {
        log.error('PRE error:', error);
      }
    });

    // Hook: POST - LLM-based fact extraction
    api.on('agent_end', async (
      event: { messages?: unknown[]; success?: boolean },
      ctx: Record<string, unknown>
    ) => {
      try {
        if (!event.messages?.length) {
          log.debug('POST: No messages in event');
          return;
        }

        const userId = extractUserId(ctx);
        if (!userId) {
          log.debug('POST: No userId extracted from context');
          return;
        }

        const messages = event.messages as Array<{ role?: string; content?: unknown }>;
        const lastUserMsg = messages.filter(m => m.role === 'user').pop();
        if (!lastUserMsg?.content) {
          log.debug('POST: No user message found');
          return;
        }

        const rawText = extractTextContent(lastUserMsg.content);
        let userText = stripMetadata(rawText);
        if (!userText || userText.length < 3) {
          log.debug('POST: User text too short or empty');
          return;
        }

        if (!anthropicApiKey) {
          log.debug('POST: No API key, skipping fact extraction');
          return;
        }

        log.debug(`POST: Extracting facts from "${userText.slice(0, 50)}..."`);

        const response = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': anthropicApiKey,
            'anthropic-version': '2023-06-01'
          },
          body: JSON.stringify({
            model: extractionModel,
            max_tokens: 300,
            messages: [{
              role: 'user',
              content: `Извлеки факты о пользователе из сообщения. Верни JSON массив объектов: {"fact":"текст","importance":0.1-1.0,"category":"FACT|PREFERENCE|EVENT"}. Если фактов нет - [].

Сообщение: "${userText}"

JSON:`
            }]
          })
        });

        if (!response.ok) {
          log.warn(`POST: LLM request failed with status ${response.status}`);
          return;
        }

        const data = await response.json() as { content?: Array<{ text?: string }> };
        const llmText = data.content?.[0]?.text || '';

        let facts: Array<{ fact: string; importance?: number; category?: string }> = [];
        try {
          const jsonMatch = llmText.match(/\[[\s\S]*\]/);
          if (jsonMatch) facts = JSON.parse(jsonMatch[0]);
        } catch (parseError) {
          log.debug('POST: Failed to parse LLM response as JSON');
          return;
        }

        if (!facts.length) {
          log.debug('POST: No facts extracted');
          return;
        }

        log.info(`POST: Extracted ${facts.length} facts for user=${userId}`);
        log.debug('POST: Facts:', facts.map(f => f.fact));

        await ensureDbInitialized(dbUrl);
        let storedCount = 0;

        for (const item of facts) {
          const factText = item.fact;
          if (!factText || factText.length < 3) continue;

          const embedding = await embed(factText);
          const existing = await searchSimilar(userId, embedding, 1, 0.9);
          if (existing.length > 0) {
            log.debug(`POST: Skipping duplicate fact: "${factText.slice(0, 30)}..."`);
            continue;
          }

          await storeMemory(userId, factText, embedding, item.importance || 0.7, item.category || 'FACT', { source: 'auto' });
          storedCount++;
        }

        if (storedCount > 0) {
          log.info(`POST: Stored ${storedCount} new facts for user=${userId}`);
        }
      } catch (error) {
        log.error('POST error:', error);
      }
    });

    log.info('Plugin registered');
  },
};

export default jarvisMemoryPlugin;
