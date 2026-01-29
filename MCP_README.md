# OM1 MCP Integration

This document describes the Model Context Protocol (MCP) integration for OM1. The integration is **bidirectional**:

- OM1 can **consume** external MCP tools (client/host mode).
- OM1 can **expose** its own actions as MCP tools (server mode).

The design is opt-in and does not affect existing behavior unless the `mcp` block is enabled in a config file.

## What You Get

- **MCP Client**: OM1 connects to external MCP servers and exposes their tools to the LLM as OpenAI-compatible tool schemas.
- **MCP Server**: OM1 exposes its actions via MCP so other hosts can call `om1_action` remotely.
- **Result Feedback**: MCP tool results are injected back into the LLM prompt as an input plugin (`MCPToolResults`).

## Architecture (High Level)

1. **Inputs** are fused into a prompt.
2. **LLM** sees native OM1 actions + MCP tools.
3. **Tool calls** are split:
   - MCP tool calls are executed via MCP client sessions.
   - Native OM1 actions go through ActionOrchestrator as usual.
4. **Tool results** are stored and injected into the next prompt.
5. **OM1 MCP Server** exposes `om1_actions` and `om1_action` to the outside.

## Configuration

Add an `mcp` block to any config file in `config/*.json5`.

```json5
"mcp": {
  "enabled": true,
  "tool_prefix": "mcp",
  "inject_results_input": true,
  "results_input_config": {
    "descriptor": "Recent MCP tool results"
  },
  "clients": [
    {
      "name": "memory",
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    {
      "name": "weather",
      "transport": "streamable-http",
      "url": "http://localhost:8001/mcp"
    }
  ],
  "server": {
    "enabled": true,
    "name": "om1",
    "transport": "streamable-http",
    "host": "0.0.0.0",
    "port": 8765,
    "path": "/mcp",
    "allow_actions": ["move", "speak"]
  }
}
```

### Notes

- `tool_prefix` must be **`mcp`** so MCP tool arguments are preserved correctly.
- `clients` entries can use `stdio` or `streamable-http` transport.
- `server` exposes OM1 actions via MCP for external clients.

## OM1 MCP Server Tools

The OM1 MCP server exposes two tools:

- `om1_actions`: list OM1 actions with their input schemas.
- `om1_action`: execute a specific OM1 action with params.

Example MCP call (conceptual):

```json
{
  "tool": "om1_action",
  "arguments": {
    "action": "speak",
    "params": {"text": "Hello"}
  }
}
```

## MCP Tool Naming

MCP client tools are registered with names like:

```
mcp__<server_name>__<tool_name>
```

Example:

```
mcp__memory__<tool_name>
```

These names appear in LLM tool calls.

## Security & Controls

For production use, always restrict exposed capabilities:

- **Client allowlist/denylist**: control which MCP tools the LLM can call.
- **Server allow_actions/deny_actions**: restrict which OM1 actions are exposed.
- Run external MCP servers with least privileges.

## Docker Notes

Docker images are unchanged. MCP adds a Python dependency (`mcp`) only. If MCP is disabled, runtime behavior is unchanged.

If you use stdio MCP servers in Docker, ensure `command` is available in the container (e.g., `npx` or local binaries).

## Troubleshooting

- **No MCP tools in LLM**: check `mcp.enabled` and that MCP client servers are reachable.
- **Tool call errors**: inspect tool allowlist/denylist and server logs.
- **Actions not exposed**: confirm server `allow_actions` or `deny_actions` settings.
- **No MCP results in prompt**: ensure `inject_results_input: true`.

## Validation Commands

```bash
uv run src/cli.py validate-config spot
uv run src/cli.py validate-config spot_modes
```

These verify schema + component availability.
