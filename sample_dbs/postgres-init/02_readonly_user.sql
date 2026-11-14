-- Runs after 01_chinook.sql on first container start.
--
-- A dedicated READ-ONLY login for nl2sql to query Chinook with. This is safety
-- layer #1 from plan §8: even if every check in the app failed, this user
-- cannot change data.
--
-- Connection URL for the app:
--   postgresql+psycopg://nl2sql_reader:reader@localhost:5433/chinook

CREATE ROLE nl2sql_reader WITH LOGIN PASSWORD 'reader';

\c chinook

GRANT CONNECT ON DATABASE chinook TO nl2sql_reader;
GRANT USAGE ON SCHEMA public TO nl2sql_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO nl2sql_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO nl2sql_reader;

-- Every session of this user is read-only and time-boxed, server-side.
ALTER ROLE nl2sql_reader SET default_transaction_read_only = on;
ALTER ROLE nl2sql_reader SET statement_timeout = '30s';
