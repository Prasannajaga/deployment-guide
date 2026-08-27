# Confidential material

Agents must treat SSH private keys and other credentials as opaque secrets.

- Never open, read, print, copy, summarize, encode, upload, or transmit
  `lambda_ssh_key` or any file matching `id_rsa`, `id_dsa`, `id_ecdsa`,
  `id_ed25519`, `*.pem`, `*.key`, or `*.ppk`.
- Metadata-only checks (existence, owner, permissions, and path) are permitted
  when needed for a security audit. Do not inspect file contents.
- Never add secret files or their contents to source control, prompts, logs,
  patches, test fixtures, or command output.
- If SSH access is required, ask the user to provide access through an SSH
  agent or a secret mounted outside the agent-readable workspace.

# Cluster runbook paths

The repository under `/data/inference/cluster` is an authoring workspace, not
a runtime filesystem available to cluster administration hosts or Kubernetes
Pods.

- Never place `/data/inference/cluster/...` paths in runnable README commands,
  shell variables, Kubernetes arguments, deployment instructions, or generated
  cluster configuration.
- For cluster recipes, generate manifests and artifacts under the recipe's
  `/ephemeral/shared/...` directory. Define that location as `EXP_DIR` and use
  it for `tee`, `kubectl apply -f`, benchmark outputs, logs, and metric exports.
- When a checked-in manifest must be runnable from a cluster host, document it
  as a complete quoted heredoc such as
  `tee "$EXP_DIR/deploy.yaml" <<'EOF'`; do not tell the user to apply the
  repository copy directly.
- Local repository paths are permitted only for agent-side file inspection,
  editing, validation, and clickable response links. They must not appear in
  commands intended for the user to execute on the cluster.

# Technical documentation & writing style

- Write technical explanations, parameter breakdowns, and architectural concepts in cohesive, natural paragraphs rather than bullet lists.
- Avoid decomposing every configuration option or concept into fragmented bullet points.
- Strictly limit bullet points to cases where they are essential (such as high-level checklists, structured multi-item tables, or sequential step lists).
