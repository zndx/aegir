-- migrate:up
-- ============================================================================
-- scheduled_tasks — the lineup KB upkeep task queue (the Gaius pattern)
-- ============================================================================
-- pg_cron (in the `postgres` DB; see aegir.lineup.cron) ENQUEUES rows here via
-- cron.schedule_in_database(..., database := 'aegir'); the ScheduledTaskProcessor
-- (folded into the gateway) consumes them by task_type and runs the handler
-- (aegir.lineup.maintain). A pg_notify on insert wakes the processor immediately
-- (LISTEN aegir_scheduled_tasks); the processor also polls as a fallback.
--
-- Notes are FILES, so the work (move scratch → archive) is done by the Python
-- handler — the enqueue→worker flavor — not a pure-SQL function.

CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id            BIGSERIAL PRIMARY KEY,
    task_type     TEXT        NOT NULL,
    payload       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    source        TEXT        NOT NULL DEFAULT 'manual',   -- pg_cron | manual | …
    status        TEXT        NOT NULL DEFAULT 'pending',  -- pending|running|completed|failed
    scheduled_for TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at    TIMESTAMPTZ,
    completed_at  TIMESTAMPTZ,
    error         TEXT,
    result        JSONB
);

-- Fast claim of due, pending work (the processor's FOR UPDATE SKIP LOCKED select).
CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_pending
    ON scheduled_tasks (scheduled_for) WHERE status = 'pending';

-- Wake the processor on every new pending task (real-time; poll is the fallback).
CREATE OR REPLACE FUNCTION notify_scheduled_task() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('aegir_scheduled_tasks', NEW.task_type);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS scheduled_tasks_notify ON scheduled_tasks;
CREATE TRIGGER scheduled_tasks_notify
    AFTER INSERT ON scheduled_tasks
    FOR EACH ROW WHEN (NEW.status = 'pending')
    EXECUTE FUNCTION notify_scheduled_task();

-- migrate:down
DROP TRIGGER IF EXISTS scheduled_tasks_notify ON scheduled_tasks;
DROP FUNCTION IF EXISTS notify_scheduled_task();
DROP TABLE IF EXISTS scheduled_tasks;
