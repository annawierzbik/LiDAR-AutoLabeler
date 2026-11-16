"""Storage, job and settings abstractions behind the web layer (see docs/adr/).

Only local implementations exist so far: FileSessionStore, LocalBlobStore, ThreadJobQueue. They
reproduce the single-process behaviour; other backends implement the same protocols.
"""
