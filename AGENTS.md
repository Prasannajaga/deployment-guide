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
