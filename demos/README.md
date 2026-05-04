# NeuronDB terminal demos (VHS)

This folder holds [VHS](https://github.com/charmbracelet/vhs) tapes used to regenerate the animated GIF in the root README.

## Files

| File | Purpose |
|------|---------|
| [neurondb-demo.tape](neurondb-demo.tape) | Records the README happy path: one-line Docker install, `psql`, extension + version + a small `vector` query, then quit. |

## Why VHS

VHS replays typed commands in a controlled terminal and exports a GIF. That keeps the README demo **repeatable** when install commands, ports, or copy change: update the tape, run the script, commit the new GIF.

## Regenerating the GIF

From the repository root:

```bash
./scripts/create-demo-gif.sh
```

Or: `make demo-gif`

Output: [docs/assets/neurondb-demo.gif](../docs/assets/neurondb-demo.gif). Details: [docs/assets/README.md](../docs/assets/README.md).

## Docker image: Hub vs local build

The tape uses the same **curl** one-liner as [README.md](../README.md), which pulls **`neurondb/neurondb:latest`** from Docker Hub by default.

`create-demo-gif.sh` first tries **`docker pull neurondb/neurondb:latest`**. If that fails (for example the tag is not published yet), the script **builds** the CPU image from [docker/neurondb/Dockerfile](../docker/neurondb/Dockerfile) and tags it **`neurondb/neurondb:latest`** locally so recording still works. End users following the README expect the Hub image; maintainers generating assets may rely on the local build path.

## README parity and port `5433`

The tape types the **exact** README install command (default container name **`neurondb`**, host port **5433**). Before recording, `create-demo-gif.sh`:

1. Starts a **throwaway** verification container **`neurondb-readme-demo`** on port **15433** to confirm the image and SQL against a real database, then removes only that container (its named volume may remain for faster reruns).
2. Refuses to run VHS if a container named **`neurondb`** already exists, or if **localhost:5433** appears occupied (when `nc` is available), so we never silently overwrite a developer’s stack.

To record on a machine that already has **`neurondb`** on 5433, stop/remove that container first or use another host.

## Updating the tape

When install URLs, default ports, credentials, or SQL smoke checks change in the README, edit **`neurondb-demo.tape`** so the typed lines stay identical to the README (except timing `Sleep` values). Re-run **`scripts/create-demo-gif.sh`** and commit **`docs/assets/neurondb-demo.gif`**.
