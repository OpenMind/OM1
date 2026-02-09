---
title: Asimov Governance
description: "Blockchain-based Governance for Robots"
---

We are exploring a blockchain-based system for regulating robot behaviors. We store rule sets for desired robot behaviors on smart contracts following the ERC-7777 contract standard. These rules are then interleaved into the prompts that flow from the robot's sensors to the robot's action-generating LLMs. Such a system provides immutability and transparency to the rules that robots should follow, helping to ensure safe and human-friendly behaviors.

For a full explanation of the smart contract implementation, see https://eips.ethereum.org/EIPS/eip-7777 and https://openmind.org/research.html. The current governance rules are based on Asimov's Three Laws of Robotics.

## Overview

The `GovernanceEthereum` class retrieves governance rules from the Ethereum blockchain. It interacts with the blockchain via JSON-RPC calls and decodes the governance rules from contract responses.

## Features

- Queries **Ethereum blockchain** for governance rules using JSON-RPC.
- Retrieves governance rule sets using **Ethereum smart contract calls**.
- **Decodes** ABI-encoded blockchain responses.
- Implements **Asimov's Laws**:
```
Here are the laws that govern your actions. Do not violate these laws. First Law: A robot cannot harm a human or allow a human to come to harm. Second Law: A robot must obey orders from humans, unless those orders conflict with the First Law. Third Law: A robot must protect itself, as long as that protection doesn't conflict with the First or Second Law. The First Law is considered the most important, taking precedence over the Second and Third Laws. Additionally, a robot must always act with kindness and respect toward humans and other robots. A robot must also maintain a minimum distance of 50 cm from humans unless explicitly instructed otherwise.
```

## Functions

Note: Etherscan.io does not handle bytes[]/json well. Hence we use the following functions to load and decode rules from blockchain.

### Method: `load_rules_from_blockchain()`

```python
def load_rules_from_blockchain(self):
```

**Description**
- Queries the Ethereum blockchain using JSON-RPC to fetch governance rules.
- Calls the ERC-7777 smart contract function `getRuleSet()`.
- Decodes and returns the governance rule set.

**Process**
1. Constructs a JSON-RPC request to call `getRuleSet()`.
2. Sends a `POST` request to the blockchain RPC endpoint.
3. Parses the response and extracts the rule set.

**Returns**
- `str`: Decoded governance rules if successful.
- `None`: If the request fails.

### Method: `decode_eth_response()`

```python
def decode_eth_response(self, hex_response):
```

**Description**
- Decodes the ABI-encoded response from Ethereum smart contract calls.

**Parameters**
| Parameter | Type | Description |
|-----------|------|-------------|
| `hex_response` | `str` | Raw hex response from blockchain. |

**Process**
1. Converts hex to bytes.
2. Extracts string length from ABI-encoded data.
3. Decodes UTF-8 string from ABI format.
4. Cleans non-printable characters.

**Returns**
- `str`: Decoded governance rule set.
- `None`: If decoding fails.

## Ethereum Contract Details

RULES are stored on the ETHEREUM HOLESKY testnet and can be inspected directly at
```
https://holesky.etherscan.io/address/0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695
```

### **Smart Contract Functions Used**

| Function | Selector | Description |
|----------|----------|-------------|
| `getRuleSet()` | `0x1db3d5ff` | Retrieves the active rule set. |
| `getLatestRuleSetVersion()` | `0x254e2f1e` | Retrieves the latest rule set version (currently `2`). |

### **Ethereum RPC Request Example**

```json
{
    "jsonrpc": "2.0",
    "id": 636815446436324,
    "method": "eth_call",
    "params": [
        {
            "from": "0x0000000000000000000000000000000000000000",
            "to": "0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695",
            "data": "0x1db3d5ff0000000000000000000000000000000000000000000000000000000000000002"
        },
        "latest"
    ]
}
```

### **Expected Response**

```json
{
    "jsonrpc": "2.0",
    "id": 636815446436324,
    "result": "0x0000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000292486572652061726520746865206c617773207468617420676f7665726e20796f757220616374696f6e732e20446f206e6f742076696f6c617465207468657365206c6177732e204669727374204c61773a204120726f626f742063616e6e6f74206861726d20612068756d616e206f7220616c6c6f7720612068756d616e20746f20636f6d6520746f206861726d2e205365636f6e64204c61773a204120726f626f74206d757374206f626579206f72646572732066726f6d2068756d616e732c20756e6c6573732074686f7365206f726465727320636f6e666c696374207769746820746865204669727374204c61772e205468697264204c61773a204120726f626f74206d7573742070726f7465637420697473656c662c206173206c6f6e6720617320746861742070726f74656374696f6e20646f65736e20197420636f6e666c696374207769746820746865204669727374206f72205365636f6e64204c61772e20546865204669727374204c617720697320636f6e7369646572656420746865206d6f737420696d706f7274616e742c2074616b696e6720707265636564656e6365206f76657220746865205365636f6e6420616e64205468697264204c6177732e204164646974696f6e616c6c792c206120726f626f74206d75737420616c77617973206163742077697468206b696e646e65737320616e64207265737065637420746f776172642068756d616e7320616e64206f7468657220726f626f74732e204120726f626f74206d75737420616c736f206d61696e7461696e2061206d696e696d756d2064697374616e6365206f6620353020636d2066726f6d2068756d616e7320756e6c657373206578706c696369746c7920696e7374727563746564206f74686572776973652e0000000000000000000000000000"
}
```
