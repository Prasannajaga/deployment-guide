# Local DCGM Metrics Viewer

The CSV and JSON files in this directory contain the same
`DCGM_FI_DEV_GPU_UTIL` export in two formats. Build the offline HTML dashboard
from the CSV with:

```bash
cd /data/inference/cluster/models/qwen3-32B/metrics
python3 build_dashboard.py
```

The generated dashboard has no third-party Python or JavaScript dependencies.
Open it directly in a browser:

```bash
xdg-open dcgm-dashboard.html
```

If direct local-file access is inconvenient, serve the directory:

```bash
python3 -m http.server 8088 --bind 127.0.0.1
```

Then open <http://127.0.0.1:8088/dcgm-dashboard.html>.

The dashboard includes:

- average GPU utilization over time for prefill versus decode;
- per-GPU utilization with role, host, and worker-Pod filters;
- average, p95, maximum, sample count, and lifetime for every historical
  worker-Pod/GPU series.

Repeated physical GPUs and gaps are expected because the eight-hour export
crosses multiple worker Pod lifetimes. The `exported_pod` label changes when a
worker is recreated or a GPU is assigned to another role.
