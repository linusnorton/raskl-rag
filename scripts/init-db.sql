-- Create the light database for BGE-M3 embeddings (if not exists)
SELECT 'CREATE DATABASE raskl_rag_light OWNER raskl'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'raskl_rag_light')\gexec
