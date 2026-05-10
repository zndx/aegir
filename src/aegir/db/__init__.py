"""Aegir relational-DB helpers.

M1 scope: just enough to run migrations at startup so downstream M2 code can
rely on a populated schema. The leaderboard itself does not touch Postgres
(reads JSON sidecars from ``outputs/runs/`` instead), so this package is
deliberately minimal and decoupled.
"""
