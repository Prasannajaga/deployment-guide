# Recover NVLS on H100 NVSwitch Nodes

Use this guide to troubleshoot NCCL workloads that fail during NVLink SHARP (NVLS) initialization.

## 1. Identify the failure

If the NCCL failure contains the following signature:

```text
NCCL WARN Failed to bind NVLink SHARP (NVLS) Multicast memory: CUDA error 802 'system not yet initialized'
RuntimeError: NCCL error: unhandled cuda error
```
Check Fabric Manager, GPU fabric registration, and the kernel log on the affected host:

```bash
sudo systemctl status nvidia-fabricmanager --no-pager -l || true

nvidia-smi -q |
  grep -i -A 2 '^    Fabric$'

sudo journalctl -k -b --no-pager |
  grep -E 'FABRIC_STATE_OUT_OF_SYNC|FABRIC_MANAGER_NOT_PRESENT' |
  tail -n 20
```

## 2. Stop GPU management clients

> [!NOTE]
>
> This procedure interrupts all GPU workloads and resets every GPU and NVSwitch on the node.

Stop all GPU applications on the node. Also stop GPU monitoring, Kubernetes reconciliation (for example, `kubelet`), and Fabric Manager:

```bash
pkill -u "$USER" -f '[w]atch.*nvidia-smi' 2>/dev/null || true
pkill -u "$USER" -x nvidia-smi 2>/dev/null || true

sudo systemctl stop nvidia-snapshot.timer 2>/dev/null || true
sudo systemctl stop nvidia-snapshot.service 2>/dev/null || true
sudo systemctl stop kubelet
sudo systemctl stop nvidia-fabricmanager 2>/dev/null || true

sleep 5
```

Check for remaining GPU management processes and processes that still have NVIDIA device files open:

```bash
pgrep -af \
  'dcgm-exporter|nv-hostengine|nvidia-device-plugin|nvidia-mig-manager|gpu-feature-discovery' \
  || true

sudo fuser -v \
  /dev/nvidiactl \
  /dev/nvidia-uvm \
  /dev/nvidia-uvm-tools \
  /dev/nvidia-modeset \
  /dev/nvidia-nvlink \
  /dev/nvidia-nvswitchctl \
  /dev/nvidia-nvswitch{0..3} \
  /dev/nvidia{0..7} 2>&1 || true
```

Verify every reported PID and send `SIGTERM` only to confirmed GPU management clients. Do not reset while either command still reports a process using the GPUs.

## 3. Reset the GPUs and NVSwitches

Run the reset without `-i` so that the full set of H100 GPUs and NVSwitches is reset:

```bash
sudo nvidia-smi -r
echo "GPU reset exit code: $?"
```

Proceed only if the exit code is `0`, which means no error, and the output reports successful resets for all eight GPUs and the NVSwitches.

## 4. Reinitialize the fabric

Restart Fabric Manager only after the reset has completed:

```bash
sudo systemctl reset-failed nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager

sleep 10

echo "=== FABRIC MANAGER ==="
systemctl is-active nvidia-fabricmanager

echo "=== FABRIC REGISTRATION ==="
nvidia-smi -q |
  grep -i -A 2 '^    Fabric$'

echo "=== NEW FABRIC ERRORS ==="
sudo journalctl -k -b --no-pager |
  grep -E 'FABRIC_STATE_OUT_OF_SYNC|FABRIC_MANAGER_NOT_PRESENT' |
  tail -n 20
```

Proceed only if Fabric Manager is `active`, all eight GPUs report
`Completed/Success`, and there are no matching kernel errors.

## 5. NVLS smoke test

Keep `kubelet` stopped while running the test to prevent Kubernetes from recreating the GPU monitoring Pods. As a smoke test, run `all_reduce_perf` from [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) with **NVLS enabled** so that a successful fallback cannot hide a broken multicast path:

```bash
NCCL_NVLS_ENABLE=1 \
NCCL_ALGO=NVLS \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,NVLS \
/usr/bin/all_reduce_perf \
  -b 8M \
  -e 128M \
  -f 2 \
  -g 8 \
  2>&1

echo "NVLS smoke exit code: $?"

sudo journalctl -k -b --no-pager |
  grep -E 'FABRIC_STATE_OUT_OF_SYNC|FABRIC_MANAGER_NOT_PRESENT' |
  tail -n 20
```

Proceed only if the exit code is `0`, the output includes `Out of bounds values : 0 OK`, there are no NCCL or CUDA failures, and there are no new matching kernel errors.

## 6. Restore the node

Restart the stopped services after the smoke test passes. If the reset or validation
fails, still run this block before further troubleshooting:

```bash
sudo systemctl reset-failed nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager
sudo systemctl start kubelet
sudo systemctl start nvidia-snapshot.timer 2>/dev/null || true

systemctl is-active nvidia-fabricmanager
systemctl is-active kubelet
systemctl is-active nvidia-snapshot.timer 2>/dev/null || true
```

After a successful smoke test, verify that the GPU Operator Pods recover.

Check [this note](https://github.com/junuxyz/mlsys-notes/blob/main/notes/distributed/recovering-nvls-on-h100.md) if you're interested in the worklog.

## References

- [NVIDIA Fabric Manager: Initializing NVSwitch and NVLink](https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html#initializing-nvswitch-and-nvlink)
- [NVIDIA System Management Interface: GPU reset](https://docs.nvidia.com/deploy/nvidia-smi/index.html#gpu-reset)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)
