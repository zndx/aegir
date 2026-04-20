// Standalone PGlite server for Aegir's CAI deployment.
// Exposes PostgreSQL wire protocol over TCP so Python (psycopg/SQLAlchemy)
// connects with a standard connection string. pgvector is bundled.
//
// Usage:
//   PGLITE_DATA_DIR=.app/pgdata PGLITE_PORT=5545 node scripts/pglite-server.mjs
//
// Port 5545 chosen to avoid conflict with CAI's platform Postgres on 5432
// (which CAI reserves for its own metastore) and with Aegir's devenv
// Postgres on 5555. Mirrors the Atelier pattern at
// ~/local/src/zndx/atelier/scripts/pglite-server.mjs; differences: default
// port (5545 vs 5440), no atelier-specific seed logic.

import { PGlite } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite/vector'
import { PGLiteSocketServer } from '@electric-sql/pglite-socket'

const DATA_DIR = process.env.PGLITE_DATA_DIR || '.app/pgdata'
const PORT = parseInt(process.env.PGLITE_PORT || '5545', 10)

// Log unhandled errors so crashes leave a trail instead of the process
// silently disappearing and leaving connect() attempts to time out.
process.on('unhandledRejection', (reason) => {
  console.error('[pglite] unhandledRejection:', reason)
})
process.on('uncaughtException', (err) => {
  console.error('[pglite] uncaughtException:', err)
  // Crash loudly — start-app.sh's wait_for_pglite will detect and fail fast.
  process.exit(1)
})

const db = await PGlite.create({
  dataDir: DATA_DIR,
  extensions: { vector },
})
console.log(`[pglite] database initialized, data at ${DATA_DIR}`)
await db.exec('CREATE EXTENSION IF NOT EXISTS vector;')
console.log(`[pglite] pgvector extension ready`)

const server = new PGLiteSocketServer({
  db,
  port: PORT,
  host: '127.0.0.1',
  // Allow a handful of concurrent connections so migrations, gateway, and
  // readiness probes don't block each other. PGlite serialises queries
  // internally so this is safe; it just keeps the TCP accept queue open.
  maxConnections: 8,
})

server.start()
  .then(() => console.log(`PGlite listening on 127.0.0.1:${PORT}, data at ${DATA_DIR}`))
  .catch(err => {
    console.error('[pglite] FATAL: server.start() failed:', err)
    process.exit(1)
  })

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, async () => {
    try {
      await server.stop()
      await db.close()
    } catch (err) {
      console.error(`[pglite] error during ${sig} shutdown:`, err)
    }
    process.exit(0)
  })
}
