# Signals peer unit (aegir.service)

Ægir joins the Signals lattice as peer id `aegir`. Process lifecycle is a
systemd oneshot under `signals.target`; the wire contract is
`zndx.engine.v1.Engine` on **:50151**.

**Full-stack doctrine:** `signals.target` start must bring the **entire**
project devenv stack (UI + gateway + capability engine + supporting services),
not a lattice-only process. Accept is lattice Status **and** product surfaces.

| Fact | Value |
|------|--------|
| Peer id | `aegir` |
| Unit | `aegir.service` (`After=signals-ready.service`) |
| Wrappers | `scripts/systemd_start.sh` / `scripts/systemd_stop.sh` |
| gRPC lattice | `:50151` — `zndx.engine.v1.Engine` (+ native + OIP) |
| Gateway | `:8091` — `http://127.0.0.1:8091/api/health` |
| Vite UI | `:5173` |
| Postgres | `:5555` — never Signals `:5455` / RustFS `:9010` |
| Capability | `instruct` |
| Status.project | `aegir` |
| Status.surfaces | `kind=primary` → product UI (`AEGIR_PRIMARY_UI` / `:5173`) |
| ServerQuery | remotes + head, configured peers, same surfaces |

## Wrappers

`systemd_start.sh` runs **`just up`** / `devenv up -d` (process-compose includes
`capability-engine` on `:50151`, gateway, vite, …) and blocks until:

1. `Engine/Status` on `:50151` (`project=aegir`)
2. Gateway health on `:8091`
3. Vite on `:5173`

`systemd_stop.sh` runs devenv processes down for **this** tree, then reclaims
any leftover `aegir.engine` on `:50151`. It does **not** call host teardown or
GPU wipe (sibling leases).

```bash
# Same path as the unit
just up
grpcurl -plaintext 127.0.0.1:50151 zndx.engine.v1.Engine/Status
curl -sf http://127.0.0.1:8091/api/health
curl -sf -o /dev/null http://127.0.0.1:5173/
just down   # or: devenv processes down
```

## Operator (from Signals)

```bash
just install-systemd --peers aegir --enable
sudo systemctl start signals.target
just lattice-ci --require aegir
```

## Platform Metaflow

When federated, platform Metaflow is SoR:

```bash
export METAFLOW_SERVICE_URL=http://127.0.0.1:30180
```

## Federation waffle

`Status.surfaces` advertises the lineup UI (never loopback). `Engine/ServerQuery`
answers remotes (`git remote -v` + HEAD), configured lattice peers, and the
same surfaces. The header waffle (`/api/aegir/v1/federation/surfaces`) lists
only engines that advertise a primary UI; LAN IP visits rebase this-host
links onto the browser Host (same as Signals / Gaius).
