"""Storage, job and settings abstractions behind the web layer (see docs/adr/).

SessionStore: FileSessionStore (default) and PgSessionStore (LAL_SESSION_STORE=postgres).
BlobStore: LocalBlobStore. JobQueue: ThreadJobQueue. Settings: LAL_* environment variables.
"""
