---
title: Manage Account and Keys
description: "Managing your OpenMind Account and API Keys"
---

Your OpenMind account and API keys are used for authentication and authorization in your applications. You can manage your account and keys via the [OpenMind portal](https://portal.openmind.org) or directly via an API. The API provides access to the full set of operations (generate, delete, account balance, and key listing).

**Base URL:** `https://api.openmind.org/api/core`

**Authentication:** All endpoints require a JWT token generated with [Clerk](https://clerk.com/). Include the token in the `Authorization` header as a Bearer token.

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api_keys/create` | Create a new API key |
| POST | `/api_keys/delete` | Delete an existing API key |
| GET | `/account/balance` | Get your account balance |
| GET | `/api_keys` | List all API keys |

## Create API Key

Create a new API key for your account. Each plan has a limit on the number of API keys you can create.

**Endpoint:** `POST /api_keys/create`

### Request

```bash
curl -X POST https://api.openmind.org/api/core/api_keys/create \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json"
```

### Response

**Success (200 OK):**

```json
{
  "message": "API key created successfully",
  "api_key": "om1_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6",
  "api_key_info": {
    "id": "a1b2c3d4e5f6g7h8",
    "user_id": "user_2abc123xyz",
    "name": "api-key-1738713600000",
    "prefix": "om1_live_",
    "total_cost": 0,
    "deleted": false,
    "created_at": "2026-02-04T10:00:00Z",
    "updated_at": "2026-02-04T10:00:00Z"
  }
}
```

**Error Responses:**

```json
// 401 Unauthorized - Missing or invalid JWT token
{
  "error": "User not found"
}

// 403 Forbidden - API key limit reached
{
  "error": "API key limit reached. Your starter plan allows up to 5 API keys. Please delete an existing key or upgrade your plan."
}

// 500 Internal Server Error
{
  "error": "Failed to create API key"
}
```

<Note>The returned `api_key` is only shown once. Store it securely as you won't be able to retrieve it again.</Note>

## Delete API Key

Mark an API key as deleted and invalidate it. This action cannot be undone.

**Endpoint:** `POST /api_keys/delete`

### Request

```bash
curl -X POST https://api.openmind.org/api/core/api_keys/delete \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "a1b2c3d4e5f6g7h8"
  }'
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | The ID of the API key to delete |

### Response

**Success (200 OK):**

```json
{
  "message": "API key deleted successfully"
}
```

**Error Responses:**

```json
// 400 Bad Request - Missing or invalid ID
{
  "error": "No data provided or invalid format"
}

{
  "error": "API key ID is required"
}

// 401 Unauthorized - Missing or invalid JWT token
{
  "error": "User not found"
}

// 404 Not Found - API key doesn't exist or doesn't belong to user
{
  "error": "API key not found"
}

// 500 Internal Server Error
{
  "error": "Failed to delete API key"
}
```

## Get Account Balance

Retrieve your current OMCU (OpenMind Compute Unit) balance and subscription details.

**Endpoint:** `GET /account/balance`

### Request

```bash
curl -X GET https://api.openmind.org/api/core/account/balance \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Response

**Success (200 OK):**

```json
{
  "plan": "pro",
  "omcu_balance": 150000,
  "monthly_unused_omcu": 50000,
  "monthly_total_omcu": 100000,
  "current_period_end": "2026-03-04T10:00:00Z",
  "cancel_at_period_end": false
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `plan` | string | Your current subscription plan (e.g., "free", "starter", "pro", "enterprise") |
| `omcu_balance` | integer | Total available OMCU credits (monthly + unused prepaid) |
| `monthly_unused_omcu` | integer | Unused OMCU credits from your monthly subscription allowance |
| `monthly_total_omcu` | integer | Total OMCU credits allocated for the current billing period |
| `current_period_end` | string \| null | ISO 8601 timestamp of when the current billing period ends |
| `cancel_at_period_end` | boolean | Whether the subscription will be cancelled at the end of the current period |

**Error Responses:**

```json
// 401 Unauthorized - Missing or invalid JWT token
{
  "error": "User not found"
}

// 404 Not Found - User account not found
{
  "error": "User not found"
}
```

<Note>
Note the following about your account balance:
- `omcu_balance` represents your total available credits, including both monthly subscription credits and any prepaid/unused credits from previous periods
- Monthly credits reset at the start of each billing period
- Unused prepaid credits do not expire and carry over between billing periods
</Note>

## List API Keys

Retrieve a list of all active API keys for your account.

**Endpoint:** `GET /api_keys`

### Request

```bash
curl -X GET https://api.openmind.org/api/core/api_keys \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Response

**Success (200 OK):**

```json
{
  "api_keys": [
    {
      "id": "a1b2c3d4e5f6g7h8",
      "user_id": "user_2abc123xyz",
      "name": "api-key-1738713600000",
      "prefix": "om1_live_",
      "total_cost": 12500,
      "deleted": false,
      "created_at": "2026-01-15T10:00:00Z",
      "updated_at": "2026-02-04T10:00:00Z"
    },
    {
      "id": "z9y8x7w6v5u4t3s2",
      "user_id": "user_2abc123xyz",
      "name": "api-key-1738800000000",
      "prefix": "om1_live_",
      "total_cost": 8750,
      "deleted": false,
      "created_at": "2026-02-01T15:30:00Z",
      "updated_at": "2026-02-04T10:00:00Z"
    }
  ]
}
```

### Response Fields

Each API key object contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the API key |
| `user_id` | string | The user ID this key belongs to |
| `name` | string | Auto-generated name for the key (format: "api-key-{timestamp}") |
| `prefix` | string | The prefix of the API key ("om1_live_" or "om1_test_") |
| `total_cost` | integer | Total OMCU credits consumed by this API key |
| `deleted` | boolean | Whether the key is deleted (always false in this response) |
| `created_at` | string | ISO 8601 timestamp of key creation |
| `updated_at` | string | ISO 8601 timestamp of last update |

**Error Responses:**

```json
// 401 Unauthorized - Missing or invalid JWT token
{
  "error": "User not found"
}

// 500 Internal Server Error
{
  "error": "Failed to fetch API keys"
}
```

<Note>
Note the following about the API keys listed:
- Only active (non-deleted) API keys are returned.
- The actual secret portion of the API key is never returned in this endpoint.
- The `hashed_key` field is stored in the database but not exposed in the API response for security.
</Note>

## Authentication

All endpoints require authentication using a JWT token issued by Clerk. Include the token in the Authorization header:

```bash
Authorization: Bearer YOUR_JWT_TOKEN
```

### Getting Your JWT Token

You can obtain your JWT token through:
1. **OpenMind Portal:** Log in at [portal.openmind.org](https://portal.openmind.org) and copy your session token
2. **Clerk SDK:** Use the Clerk client library to authenticate and retrieve the session token programmatically

### Example with Token

```bash
# Set your token as an environment variable
export OPENMIND_JWT_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Use it in your requests
curl -X GET https://api.openmind.org/api/core/account/balance \
  -H "Authorization: Bearer $OPENMIND_JWT_TOKEN"
```

## Error Handling

All endpoints follow consistent error response patterns:

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input parameters |
| 401 | Unauthorized - Missing or invalid JWT token |
| 403 | Forbidden - Action not allowed (e.g., rate limit reached) |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error - Server-side error |

### Error Response Format

```json
{
  "error": "Descriptive error message"
}
```

## Rate Limits

API key operations may be subject to rate limits depending on your subscription plan. If you exceed the rate limit, you'll receive a `429 Too Many Requests` response.
