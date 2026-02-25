# jarvis-memory

OpenClaw plugin for long-term memory with pgvector and local embeddings.

## Features

- **Semantic search** using pgvector (cosine similarity)
- **Local embeddings** via transformers.js (paraphrase-multilingual-MiniLM-L12-v2)
- **No API costs** for embeddings - runs entirely locally
- **Russian language support** (multilingual model)
- **Automatic eviction** of low-importance memories
- **Text search fallback** for proper nouns

## Tools

| Tool | Description |
|------|-------------|
| `memory_search` | Search memories using semantic similarity |
| `memory_store` | Store a new memory |
| `memory_forget` | Delete memories |

## Installation

```bash
# In your OpenClaw extensions directory
git clone https://github.com/Danny-sth/jarvis-memory.git
cd jarvis-memory
npm install
npm run build
```

## Configuration

Add to your `openclaw.json`:

```json
{
  "plugins": {
    "jarvis-memory": {
      "enabled": true,
      "config": {
        "databaseUrl": "postgresql://user:pass@localhost:5432/jarvis"
      }
    }
  }
}
```

Or set environment variable:

```bash
export JARVIS_DATABASE_URL="postgresql://user:pass@localhost:5432/jarvis"
```

## Database Schema

The plugin automatically creates the required table:

```sql
CREATE TABLE jarvis_memory (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(100) NOT NULL,
  content TEXT NOT NULL,
  embedding vector(384),
  importance FLOAT DEFAULT 0.5,
  memory_type VARCHAR(50) DEFAULT 'FACT',
  metadata JSONB DEFAULT '{}',
  access_count INTEGER DEFAULT 0,
  last_accessed_at TIMESTAMP DEFAULT NOW(),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

## Memory Types

- `FACT` - General facts about the user
- `PREFERENCE` - User preferences (likes, dislikes)
- `OPINION` - User opinions
- `EVENT` - Events, dates, appointments
- `CONTEXT` - Conversation context
