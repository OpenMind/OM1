---
title: OTA Update Manager
description: "Over-the-Air Update Manager Plugin for OM1"
icon: arrow-up-circle
---

## Overview

The OTA Update Manager is a background plugin for OM1 that periodically checks for software updates and applies them automatically or notifies the user when a new version is available. It integrates with the OpenMind OMCU token system to support paid updates.

This document describes the plugin architecture, the endpoints that need to be implemented on the OpenMind server side, and the parts that require further input from the OpenMind team.

---

## Plugin Location

| File | Path |
|------|------|
| Provider | `src/providers/ota_provider.py` |
| Background Plugin | `src/backgrounds/plugins/ota_update_manager.py` |
| Test Config | `config/ota_test.json5` |
| Provider Tests | `tests/providers/test_ota_provider.py` |
| Plugin Tests | `tests/backgrounds/plugins/test_ota_update_manager.py` |

---

## How It Works

1. The background plugin runs in its own thread and periodically calls `OTAProvider.check_for_updates()`.
2. If an update is available and it is a paid update, it checks the user's OMCU balance via `GET /account/balance`.
3. If the balance is sufficient, it downloads the package and verifies the SHA-256 hash.
4. If `auto_update=True`, it applies the update. On success, it records the OMCU transaction. On failure, it rolls back.
5. If `auto_update=False`, it logs that the update is ready and waits for manual intervention.

---

## Configuration

The plugin is configured via the runtime config file (e.g., `config/ota_test.json5`):
```json5
{
  backgrounds: [
    {
      type: "OTAUpdateManager",
      config: {
        check_interval: 3600,       // Seconds between update checks (default: 3600)
        auto_update: false,          // Set true to apply updates automatically
        update_url: "https://api.openmind.org/api/core", // Production URL
        require_balance_check: true, // Check OMCU balance before paid updates
      },
    },
  ],
}
```

---

## Authentication

All requests use a JWT token from Clerk as a Bearer token in the `Authorization` header.
```
Authorization: Bearer YOUR_JWT_TOKEN
```

For details on obtaining a JWT token, see:
`docs/api-reference/endpoints/account_and_key_management.md`

> **Action required (OpenMind team):** Please provide the mechanism for obtaining the JWT token programmatically via the Clerk SDK so it can be integrated into the plugin.

---

## Existing Endpoint Used

### Get Account Balance

This endpoint is already available and is used to check OMCU balance before applying a paid update.

**Endpoint:** `GET /account/balance`
**Base URL:** `https://api.openmind.org/api/core`

**Response field used by plugin:**

| Field | Type | Description |
|-------|------|-------------|
| `omcu_balance` | integer | Total available OMCU credits |

For full documentation see:
`docs/api-reference/endpoints/account_and_key_management.md`

---

## Endpoints Required from OpenMind Team

The following endpoints are used by the plugin but have not yet been implemented on the server side. These need to be created by the OpenMind team.

### 1. Check for Updates

**Endpoint:** `GET /api/updates/latest`

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `current_version` | string | The current installed version of OM1 |

**Expected Response (200 OK):**
```json
{
  "version": "2.0.0",
  "price": 100,
  "package_url": "https://example.com/om1_update_2.0.0.zip",
  "sha256": "abc123def456..."
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Latest available version |
| `price` | integer | Cost in OMCU (0 = free update) |
| `package_url` | string | Direct URL to download the update package |
| `sha256` | string | SHA-256 hash of the package for integrity verification |

> If the current version is already the latest, return the same version string so the plugin knows no update is needed.

---

### 2. Record Transaction

**Endpoint:** `POST /api/transactions`

Called after a successful paid update to debit the OMCU balance.

**Request Body:**
```json
{
  "amount": 100,
  "description": "OM1 update to version 2.0.0",
  "timestamp": 1738713600.0
}
```

**Request Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `amount` | float | Amount of OMCU to debit |
| `description` | string | Human-readable description of the transaction |
| `timestamp` | float | Unix timestamp of the transaction |

**Expected Response:**
```json
// 200 OK or 201 Created
{
  "message": "Transaction recorded successfully"
}
```

---

## Parts Requiring Further Implementation

The following methods in `src/providers/ota_provider.py` are currently stubs and need to be implemented based on the OM1 robot architecture:

### `apply_update(version)`

Currently simulates installation with `time.sleep(2)`. Needs to be replaced with actual logic such as:
- Backup current installation
- Extract and replace files from the downloaded `.zip` package
- Trigger a system restart if required

### `rollback()`

Currently simulates rollback with `time.sleep(1)`. Needs to be replaced with actual logic such as:
- Restore the backed-up installation
- Verify the restored version is working

---

## Running Tests
```bash
# Run provider tests
pytest tests/providers/test_ota_provider.py -v

# Run background plugin tests
pytest tests/backgrounds/plugins/test_ota_update_manager.py -v

# Run all OTA tests with coverage
pytest tests/providers/test_ota_provider.py tests/backgrounds/plugins/test_ota_update_manager.py -v --cov=src/providers/ota_provider --cov=src/backgrounds/plugins/ota_update_manager
```

---

## Summary of Actions Required from OpenMind Team

| # | Action | Priority |
|---|--------|----------|
| 1 | Implement `GET /api/updates/latest` endpoint | High |
| 2 | Implement `POST /api/transactions` endpoint | High |
| 3 | Provide JWT token integration via Clerk SDK | High |
| 4 | Implement `apply_update()` logic based on robot architecture | Medium |
| 5 | Implement `rollback()` logic based on robot architecture | Medium |
