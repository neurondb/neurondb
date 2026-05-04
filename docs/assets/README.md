# README assets

## Terminal demo GIF

| File | Description |
|------|-------------|
| [neurondb-demo.gif](neurondb-demo.gif) | Short terminal recording for the root [README.md](../../README.md): Docker install, `psql`, and NeuronDB smoke SQL. |

### Regenerate

From the repository root:

```bash
./scripts/create-demo-gif.sh
```

Requires **Docker**, **psql**, and **[VHS](https://github.com/charmbracelet/vhs)**. The script prints install hints for VHS if it is missing (nothing is installed silently).

The GIF should stay **under about 10 MB** for GitHub. If regeneration produces a larger file, shorten [demos/neurondb-demo.tape](../../demos/neurondb-demo.tape) or reduce sleeps/typing length.

See [demos/README.md](../../demos/README.md) for flow, Hub vs local image behavior, and README parity notes.
