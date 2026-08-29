#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from the exported DCGM CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean


DEFAULT_INPUT = "dcgm-gpu-util-last-8h.csv"
DEFAULT_OUTPUT = "dcgm-dashboard.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline DCGM GPU-utilization dashboard."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name(DEFAULT_INPUT),
        help=f"DCGM sample CSV (default: {DEFAULT_INPUT} beside this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(DEFAULT_OUTPUT),
        help=f"Generated HTML file (default: {DEFAULT_OUTPUT} beside this script)",
    )
    return parser.parse_args()


def parse_timestamp(value: str) -> int:
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return int(parsed.timestamp() * 1000)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def role_for_pod(pod: str) -> str:
    if "prefillworker" in pod:
        return "prefill"
    if "decodeworker" in pod:
        return "decode"
    return "other"


def iso_utc(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def load_data(path: Path) -> dict[str, object]:
    series: dict[tuple[str, str, str, str, str], list[list[float]]] = defaultdict(list)
    role_buckets: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_values: list[float] = []
    all_times: list[int] = []

    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "timestamp_utc",
            "hostname",
            "worker_pod",
            "gpu",
            "uuid",
            "gpu_util_percent",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            timestamp = parse_timestamp(row["timestamp_utc"])
            utilization = float(row["gpu_util_percent"])
            pod = row["worker_pod"]
            role = role_for_pod(pod)
            key = (role, row["hostname"], pod, row["gpu"], row["uuid"])
            series[key].append([timestamp, utilization])
            role_buckets[role][timestamp].append(utilization)
            all_values.append(utilization)
            all_times.append(timestamp)

    if not all_values:
        raise ValueError(f"No samples found in {path}")

    gpu_series = []
    for (role, host, pod, gpu, uuid), points in sorted(series.items()):
        points.sort(key=lambda point: point[0])
        values = [point[1] for point in points]
        gpu_series.append(
            {
                "id": f"{host}|{pod}|{gpu}|{uuid}",
                "role": role,
                "host": host,
                "pod": pod,
                "gpu": gpu,
                "uuid": uuid,
                "shortUuid": uuid.removeprefix("GPU-")[:8],
                "label": f"{role} · {host.split('-')[1]} · GPU {gpu} · {pod.rsplit('-', 1)[-1]}",
                "samples": len(points),
                "average": fmean(values),
                "p95": percentile(values, 0.95),
                "minimum": min(values),
                "maximum": max(values),
                "start": points[0][0],
                "end": points[-1][0],
                "points": points,
            }
        )

    role_series = []
    for role, time_buckets in sorted(role_buckets.items()):
        points = [
            [timestamp, fmean(values)]
            for timestamp, values in sorted(time_buckets.items())
        ]
        role_series.append(
            {
                "id": role,
                "label": f"{role.title()} average",
                "role": role,
                "points": points,
            }
        )

    active_values = [value for value in all_values if value > 0]
    return {
        "source": path.name,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start": min(all_times),
        "end": max(all_times),
        "startIso": iso_utc(min(all_times)),
        "endIso": iso_utc(max(all_times)),
        "samples": len(all_values),
        "streams": len(gpu_series),
        "physicalGpus": len({item[4] for item in series}),
        "pods": len({item[2] for item in series}),
        "overallAverage": fmean(all_values),
        "activeAverage": fmean(active_values) if active_values else 0,
        "maximum": max(all_values),
        "roleSeries": role_series,
        "gpuSeries": gpu_series,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Qwen3-32B DCGM GPU Utilization</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07111f;
      --panel: #0d1b2d;
      --panel-2: #10243a;
      --border: #213b55;
      --text: #e7f0f8;
      --muted: #8fa7bb;
      --prefill: #35d0ba;
      --decode: #ffb454;
      --accent: #70b7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 15% 0%, #12314b 0, var(--bg) 34rem);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, sans-serif;
    }
    main { width: min(1500px, calc(100% - 32px)); margin: 28px auto 60px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: end; }
    h1 { margin: 0; font-size: clamp(24px, 3vw, 38px); letter-spacing: -.03em; }
    h2 { margin: 0 0 14px; font-size: 18px; }
    p { color: var(--muted); margin: 7px 0 0; }
    code { color: #b9ddff; }
    .badge { padding: 7px 11px; border: 1px solid var(--border); border-radius: 999px; color: var(--muted); white-space: nowrap; }
    .cards { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 12px; margin: 22px 0; }
    .card, .panel { background: linear-gradient(145deg, rgba(16,36,58,.96), rgba(10,24,41,.96)); border: 1px solid var(--border); box-shadow: 0 18px 45px rgba(0,0,0,.18); }
    .card { padding: 15px 17px; border-radius: 12px; }
    .card span { display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .card strong { display: block; margin-top: 5px; font-size: 23px; }
    .panel { padding: 18px; border-radius: 14px; margin-top: 14px; }
    .chart-shell { position: relative; min-height: 390px; }
    canvas { width: 100%; height: 390px; display: block; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 14px; margin-top: 10px; color: var(--muted); font-size: 12px; }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 18px; height: 3px; border-radius: 2px; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }
    label { color: var(--muted); font-size: 12px; }
    select { display: block; min-width: 190px; margin-top: 4px; padding: 8px 10px; border: 1px solid var(--border); border-radius: 8px; background: #091727; color: var(--text); }
    .tooltip { display: none; position: absolute; pointer-events: none; z-index: 5; max-width: 420px; padding: 9px 11px; border: 1px solid var(--border); border-radius: 8px; background: rgba(5,14,25,.96); box-shadow: 0 10px 35px rgba(0,0,0,.4); color: var(--text); font-size: 12px; }
    .table-wrap { overflow: auto; max-height: 580px; border: 1px solid var(--border); border-radius: 9px; }
    table { width: 100%; border-collapse: collapse; white-space: nowrap; }
    th, td { padding: 9px 11px; border-bottom: 1px solid rgba(33,59,85,.65); text-align: left; }
    th { position: sticky; top: 0; background: #10243a; color: #bcd0df; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr:hover td { background: rgba(112,183,255,.055); }
    .role { padding: 3px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; }
    .role.prefill { background: rgba(53,208,186,.13); color: var(--prefill); }
    .role.decode { background: rgba(255,180,84,.13); color: var(--decode); }
    .note { padding: 11px 13px; margin-top: 12px; border-left: 3px solid var(--accent); background: rgba(112,183,255,.06); color: var(--muted); }
    @media (max-width: 900px) { .cards { grid-template-columns: repeat(2, 1fr); } header { align-items: start; flex-direction: column; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Qwen3-32B · DCGM GPU utilization</h1>
      <p id="range"></p>
    </div>
    <div class="badge">Offline dashboard · UTC</div>
  </header>

  <section class="cards" id="cards"></section>

  <section class="panel">
    <h2>Prefill vs decode average</h2>
    <p>Mean utilization across the GPUs attributed to each role at every scrape.</p>
    <div class="chart-shell">
      <canvas id="roleChart"></canvas>
      <div class="tooltip" id="roleTooltip"></div>
    </div>
    <div class="legend" id="roleLegend"></div>
  </section>

  <section class="panel">
    <h2>Per-GPU utilization</h2>
    <div class="controls">
      <label>Role<select id="roleFilter"><option value="all">All roles</option><option value="prefill">Prefill</option><option value="decode">Decode</option></select></label>
      <label>Host<select id="hostFilter"><option value="all">All hosts</option></select></label>
      <label>Worker Pod<select id="podFilter"><option value="all">All Pod lifetimes</option></select></label>
    </div>
    <div class="chart-shell">
      <canvas id="gpuChart"></canvas>
      <div class="tooltip" id="gpuTooltip"></div>
    </div>
    <div class="legend" id="gpuLegend"></div>
    <div class="note">Gaps and repeated GPU IDs are expected when worker Pods restart or a physical GPU changes role. Select a Pod to isolate one worker lifetime.</div>
  </section>

  <section class="panel">
    <h2>Series summary</h2>
    <p>One row per worker-Pod/GPU label set. Average includes idle samples.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Role</th><th>Host</th><th>Worker Pod</th><th>GPU</th><th>UUID</th><th>Samples</th><th>Average</th><th>P95</th><th>Maximum</th><th>First sample</th><th>Last sample</th></tr></thead>
      <tbody id="summaryBody"></tbody>
    </table></div>
  </section>
</main>

<script>
const DATA = __DASHBOARD_DATA__;
const COLORS = ['#70b7ff','#35d0ba','#ffb454','#f071b5','#a78bfa','#f56f6f','#9dde58','#5ed1f2','#ffdc73','#c98cff','#60d394','#ff8c61'];
const ROLE_COLORS = {prefill:'#35d0ba', decode:'#ffb454', other:'#70b7ff'};
const $ = id => document.getElementById(id);
const utc = ms => new Date(ms).toISOString().replace('.000Z','Z');
const pct = value => `${value.toFixed(1)}%`;

$('range').textContent = `${DATA.startIso} → ${DATA.endIso} · source: ${DATA.source}`;
const cards = [
  ['Samples', DATA.samples.toLocaleString()],
  ['Physical GPUs', DATA.physicalGpus],
  ['Worker Pods', DATA.pods],
  ['Historical streams', DATA.streams],
  ['Overall average', pct(DATA.overallAverage)],
  ['Observed maximum', pct(DATA.maximum)],
];
$('cards').innerHTML = cards.map(([label,value]) => `<div class="card"><span>${label}</span><strong>${value}</strong></div>`).join('');

function nearestPoint(points, target) {
  let low=0, high=points.length-1;
  while (low < high) { const mid=Math.floor((low+high)/2); if (points[mid][0] < target) low=mid+1; else high=mid; }
  const a=points[low], b=points[Math.max(0,low-1)];
  return Math.abs(a[0]-target) < Math.abs(b[0]-target) ? a : b;
}

function drawChart(canvas, tooltip, series) {
  const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio || 1;
  const width=Math.max(640, rect.width), height=390;
  canvas.width=width*dpr; canvas.height=height*dpr;
  const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr);
  const pad={left:54,right:18,top:18,bottom:42};
  const pw=width-pad.left-pad.right, ph=height-pad.top-pad.bottom;
  const minX=DATA.start, maxX=DATA.end, x=xv=>pad.left+(xv-minX)/(maxX-minX)*pw, y=yv=>pad.top+(100-yv)/100*ph;
  ctx.font='12px system-ui'; ctx.lineWidth=1;
  for (let tick=0; tick<=100; tick+=20) {
    const py=y(tick); ctx.strokeStyle='rgba(143,167,187,.16)'; ctx.beginPath(); ctx.moveTo(pad.left,py); ctx.lineTo(width-pad.right,py); ctx.stroke();
    ctx.fillStyle='#8fa7bb'; ctx.textAlign='right'; ctx.fillText(`${tick}%`,pad.left-9,py+4);
  }
  for (let i=0;i<=6;i++) {
    const tx=minX+(maxX-minX)*i/6, px=x(tx);
    ctx.strokeStyle='rgba(143,167,187,.1)'; ctx.beginPath(); ctx.moveTo(px,pad.top); ctx.lineTo(px,height-pad.bottom); ctx.stroke();
    ctx.fillStyle='#8fa7bb'; ctx.textAlign=i===0?'left':i===6?'right':'center'; ctx.fillText(new Date(tx).toISOString().slice(11,16),px,height-16);
  }
  series.forEach((item,index)=>{
    ctx.strokeStyle=item.color || COLORS[index%COLORS.length]; ctx.lineWidth=1.7; ctx.globalAlpha=.9; ctx.beginPath();
    let previous=null;
    item.points.forEach(point=>{
      const px=x(point[0]), py=y(point[1]);
      if (!previous || point[0]-previous[0] > 90000) ctx.moveTo(px,py); else ctx.lineTo(px,py);
      previous=point;
    });
    ctx.stroke();
  });
  ctx.globalAlpha=1;

  canvas.onmousemove = event => {
    const box=canvas.getBoundingClientRect();
    const mx=event.clientX-box.left, target=minX+Math.max(0,Math.min(1,(mx-pad.left)/pw))*(maxX-minX);
    const rows=series.map(item=>[item,nearestPoint(item.points,target)]).filter(([,point])=>Math.abs(point[0]-target)<=60000).sort((a,b)=>b[1][1]-a[1][1]);
    if (!rows.length) { tooltip.style.display='none'; return; }
    tooltip.innerHTML=`<strong>${utc(rows[0][1][0])}</strong><br>`+rows.slice(0,12).map(([item,point])=>`<span style="color:${item.color}">●</span> ${item.label}: ${pct(point[1])}`).join('<br>');
    tooltip.style.display='block'; tooltip.style.left=`${Math.min(mx+14,width-420)}px`; tooltip.style.top='18px';
  };
  canvas.onmouseleave=()=>tooltip.style.display='none';
}

function renderLegend(target, series) {
  target.innerHTML=series.map(item=>`<span class="legend-item"><span class="swatch" style="background:${item.color}"></span>${item.label}</span>`).join('');
}

const roleSeries=DATA.roleSeries.map(item=>({...item,color:ROLE_COLORS[item.role]}));
function drawRoles(){ drawChart($('roleChart'),$('roleTooltip'),roleSeries); renderLegend($('roleLegend'),roleSeries); }

const roleFilter=$('roleFilter'), hostFilter=$('hostFilter'), podFilter=$('podFilter');
[...new Set(DATA.gpuSeries.map(item=>item.host))].sort().forEach(host=>hostFilter.add(new Option(host,host)));

function populatePods() {
  const selected=podFilter.value;
  podFilter.innerHTML='<option value="all">All Pod lifetimes</option>';
  [...new Set(DATA.gpuSeries.filter(item=>(roleFilter.value==='all'||item.role===roleFilter.value)&&(hostFilter.value==='all'||item.host===hostFilter.value)).map(item=>item.pod))].sort().forEach(pod=>podFilter.add(new Option(pod,pod)));
  if ([...podFilter.options].some(option=>option.value===selected)) podFilter.value=selected;
}

function filteredSeries() {
  return DATA.gpuSeries.filter(item=>(roleFilter.value==='all'||item.role===roleFilter.value)&&(hostFilter.value==='all'||item.host===hostFilter.value)&&(podFilter.value==='all'||item.pod===podFilter.value)).map((item,index)=>({...item,color:ROLE_COLORS[item.role] || COLORS[index%COLORS.length]}));
}

function renderGpuView() {
  const series=filteredSeries();
  drawChart($('gpuChart'),$('gpuTooltip'),series);
  renderLegend($('gpuLegend'),series);
  $('summaryBody').innerHTML=series.slice().sort((a,b)=>b.average-a.average).map(item=>`<tr><td><span class="role ${item.role}">${item.role}</span></td><td>${item.host}</td><td>${item.pod}</td><td class="num">${item.gpu}</td><td><code>${item.shortUuid}</code></td><td class="num">${item.samples}</td><td class="num">${pct(item.average)}</td><td class="num">${pct(item.p95)}</td><td class="num">${pct(item.maximum)}</td><td>${utc(item.start)}</td><td>${utc(item.end)}</td></tr>`).join('');
}

roleFilter.onchange=()=>{populatePods();renderGpuView();}; hostFilter.onchange=()=>{populatePods();renderGpuView();}; podFilter.onchange=renderGpuView;
window.onresize=()=>{drawRoles();renderGpuView();};
populatePods(); drawRoles(); renderGpuView();
</script>
</body>
</html>
'''


def main() -> None:
    args = parse_args()
    data = load_data(args.input.resolve())
    rendered = HTML_TEMPLATE.replace(
        "__DASHBOARD_DATA__", json.dumps(data, separators=(",", ":"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Dashboard: {args.output.resolve()}")
    print(
        "Samples: {samples:,}; physical GPUs: {physicalGpus}; "
        "historical streams: {streams}".format(**data)
    )


if __name__ == "__main__":
    main()
