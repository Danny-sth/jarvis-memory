/**
 * Database module for PostgreSQL + pgvector
 */

import postgres from 'postgres';
import { DIMENSIONS } from './embeddings.js';

export interface MemoryEntry {
  id: number;
  user_id: string;
  content: string;
  embedding?: number[];
  importance: number;
  memory_type: string;
  metadata: Record<string, unknown>;
  access_count: number;
  last_accessed_at: Date;
  created_at: Date;
  updated_at: Date;
}

export interface SearchResult extends MemoryEntry {
  similarity: number;
}

let sql: postgres.Sql | null = null;

/**
 * Initialize database connection
 */
export function initDatabase(connectionString: string): void {
  sql = postgres(connectionString, {
    max: 10,
    idle_timeout: 30,
    connect_timeout: 10,
  });
  console.log('[jarvis-memory] Database connection initialized');
}

/**
 * Get database connection
 */
function getDb(): postgres.Sql {
  if (!sql) {
    throw new Error('Database not initialized. Call initDatabase first.');
  }
  return sql;
}

/**
 * Create tables if not exist
 */
export async function ensureSchema(): Promise<void> {
  const db = getDb();

  await db`CREATE EXTENSION IF NOT EXISTS vector`;

  // Use unsafe for DDL with dynamic dimensions (not user input, safe)
  await db.unsafe(`
    CREATE TABLE IF NOT EXISTS jarvis_memory (
      id SERIAL PRIMARY KEY,
      user_id VARCHAR(100) NOT NULL,
      content TEXT NOT NULL,
      embedding vector(${DIMENSIONS}),
      importance FLOAT DEFAULT 0.5,
      memory_type VARCHAR(50) DEFAULT 'FACT',
      metadata JSONB DEFAULT '{}',
      access_count INTEGER DEFAULT 0,
      last_accessed_at TIMESTAMP DEFAULT NOW(),
      created_at TIMESTAMP DEFAULT NOW(),
      updated_at TIMESTAMP DEFAULT NOW()
    )
  `);

  // Create indexes
  await db`
    CREATE INDEX IF NOT EXISTS idx_jarvis_memory_user_id
    ON jarvis_memory(user_id)
  `;

  await db`
    CREATE INDEX IF NOT EXISTS idx_jarvis_memory_embedding
    ON jarvis_memory
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
  `;

  await db`
    CREATE INDEX IF NOT EXISTS idx_jarvis_memory_importance
    ON jarvis_memory(importance DESC)
  `;

  console.log('[jarvis-memory] Database schema ensured');
}

/**
 * Search similar memories using vector similarity
 */
export async function searchSimilar(
  userId: string,
  embedding: number[],
  limit: number = 10,
  threshold: number = 0.7
): Promise<SearchResult[]> {
  const db = getDb();
  const vectorStr = '[' + embedding.join(',') + ']';

  const results = await db<SearchResult[]>`
    SELECT
      id, user_id, content, importance, memory_type,
      metadata, access_count, last_accessed_at, created_at, updated_at,
      1 - (embedding <=> ${vectorStr}::vector) as similarity
    FROM jarvis_memory
    WHERE user_id = ${userId}
      AND 1 - (embedding <=> ${vectorStr}::vector) >= ${threshold}
    ORDER BY similarity DESC
    LIMIT ${limit}
  `;

  // Update access count for retrieved memories
  if (results.length > 0) {
    const ids = results.map(r => r.id);
    await db`
      UPDATE jarvis_memory
      SET access_count = access_count + 1,
          last_accessed_at = NOW()
      WHERE id = ANY(${ids})
    `;
  }

  return results;
}

/**
 * Get ALL memories for a user (no filtering, no limit)
 */
export async function getAllMemories(userId: string): Promise<MemoryEntry[]> {
  const db = getDb();

  return await db<MemoryEntry[]>`
    SELECT
      id, user_id, content, importance, memory_type,
      metadata, access_count, last_accessed_at, created_at, updated_at
    FROM jarvis_memory
    WHERE user_id = ${userId}
    ORDER BY importance DESC, created_at DESC
  `;
}

/**
 * Full-text search fallback for proper nouns
 */
export async function searchText(
  userId: string,
  query: string,
  limit: number = 10
): Promise<MemoryEntry[]> {
  const db = getDb();

  return await db<MemoryEntry[]>`
    SELECT
      id, user_id, content, importance, memory_type,
      metadata, access_count, last_accessed_at, created_at, updated_at
    FROM jarvis_memory
    WHERE user_id = ${userId}
      AND content ILIKE ${'%' + query + '%'}
    ORDER BY importance DESC, updated_at DESC
    LIMIT ${limit}
  `;
}

/**
 * Store a new memory
 */
export async function storeMemory(
  userId: string,
  content: string,
  embedding: number[],
  importance: number,
  memoryType: string,
  metadata: Record<string, unknown> = {}
): Promise<number> {
  const db = getDb();
  const vectorStr = '[' + embedding.join(',') + ']';

  const result = await db<{ id: number }[]>`
    INSERT INTO jarvis_memory (user_id, content, embedding, importance, memory_type, metadata)
    VALUES (
      ${userId},
      ${content},
      ${vectorStr}::vector,
      ${Math.max(0, Math.min(1, importance))},
      ${memoryType},
      ${JSON.stringify(metadata)}
    )
    RETURNING id
  `;

  return result[0].id;
}

/**
 * Delete memory by ID
 */
export async function deleteMemory(id: number): Promise<boolean> {
  const db = getDb();

  const result = await db`
    DELETE FROM jarvis_memory WHERE id = ${id}
  `;

  return result.count > 0;
}

/**
 * Delete memories by user and optional filter
 */
export async function forgetMemories(
  userId: string,
  query?: string
): Promise<number> {
  const db = getDb();

  if (query) {
    const result = await db`
      DELETE FROM jarvis_memory
      WHERE user_id = ${userId}
        AND content ILIKE ${'%' + query + '%'}
    `;
    return result.count;
  } else {
    const result = await db`
      DELETE FROM jarvis_memory
      WHERE user_id = ${userId}
    `;
    return result.count;
  }
}

/**
 * Get memory count for user
 */
export async function countMemories(userId: string): Promise<number> {
  const db = getDb();

  const result = await db<{ count: string }[]>`
    SELECT COUNT(*) as count FROM jarvis_memory WHERE user_id = ${userId}
  `;

  return parseInt(result[0].count, 10);
}

/**
 * Evict lowest importance memories
 */
export async function evictLowest(userId: string, count: number): Promise<number> {
  const db = getDb();

  const result = await db`
    DELETE FROM jarvis_memory
    WHERE id IN (
      SELECT id FROM jarvis_memory
      WHERE user_id = ${userId}
      ORDER BY importance ASC, access_count ASC
      LIMIT ${count}
    )
  `;

  return result.count;
}

/**
 * Close database connection
 */
export async function closeDatabase(): Promise<void> {
  if (sql) {
    await sql.end();
    sql = null;
    console.log('[jarvis-memory] Database connection closed');
  }
}
