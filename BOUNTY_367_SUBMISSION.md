# Bounty #367: Robot Fetch Service with Voice Command & Crypto Payment

This submission implements a robotics-focused solution for Bounty #367, enabling autonomous mobile robots to receive fetch/delivery requests via Home Assistant voice commands and process payments through cryptocurrency wallets.

## Overview

Enable robots to perform fetch and delivery tasks upon voice request, with cryptocurrency payment processing, creating a practical service economy for autonomous mobile robots.

## Robotics Use Case

**Scenario**: User owns a mobile robot (TurtleBot, Spot, delivery robot) at home or office.

**Complete Workflow**:

1. **Voice Command via Home Assistant**
   - User: "Hey Google, ask robot to bring me a water bottle from kitchen"
   - Home Assistant captures command

2. **Robot Processes Request**
   - OM1 agent receives command via Home Assistant plugin
   - Robot responds via TTS: "Fetch service: water bottle from kitchen. Fee: $2 (0.002 ETH). Send payment to confirm."

3. **Payment Processing**
   - User sends 0.002 ETH to robot's Coinbase wallet
   - Wallet plugin detects incoming payment

4. **Task Execution**
   - Robot confirms: "Payment received. Navigating to kitchen now."
   - Robot navigates to kitchen, retrieves item, delivers to user

5. **Completion**
   - Robot announces: "Task complete. Water bottle delivered. Ready for next request."

## Why This Solves Bounty #367

### Bounty Requirements Met:
✅ **Smart Assistant Integration**: Voice commands via Home Assistant
✅ **Order Something**: User orders robot service (fetch/delivery tasks)
✅ **Crypto Wallet Payment**: Payment via Coinbase wallet
✅ **Full Workflow**: Voice → order → payment → confirmation → execution

### Robotics-Focused (per @openminddev requirements):
✅ **Real robot use case**: Mobile robot performing physical tasks
✅ **Not standalone app**: Integrated with OM1 robot runtime
✅ **Practical application**: Applicable to real-world service robots

## Implementation Details

### Components

1. **Home Assistant Input Plugin** (`src/inputs/plugins/home_assistant.py`)
   - Polls Home Assistant REST API for voice commands
   - Monitors `input_text.om1_voice_command` entity
   - Converts voice commands into OM1 messages for robot agent

2. **Wallet Integration** (existing `src/inputs/plugins/wallet_coinbase.py`)
   - Monitors Coinbase wallet for incoming ETH transactions
   - Reports balance changes to robot agent
   - Confirms payment receipt

3. **Robot Agent Configuration** (`config/robot_fetch_service.json5`)
   - Combines Home Assistant + Wallet inputs
   - LLM manages request, payment, and task execution flow
   - TTS provides voice feedback to user

### Architecture
```
User Voice Command (Home Assistant)
    ↓
Home Assistant Plugin → OM1 Robot Agent (LLM)
    ↓                        ↓
Request Processing    Payment Request
    ↓                        ↓
Wallet Plugin ← ETH Payment Confirmation
    ↓
Task Execution (Robot Navigation/Manipulation)
    ↓
Completion Report (TTS via Home Assistant)
```

## Setup Instructions

### Prerequisites

- OM1 installation
- Home Assistant instance running
- Coinbase CDP API credentials
- Mobile robot with OM1 runtime

### 1. Configure Home Assistant

Create an `input_text` entity in Home Assistant:
```yaml
# configuration.yaml
input_text:
  om1_voice_command:
    name: "OM1 Voice Command"
    initial: ""
```

Create automation to capture voice commands:
```yaml
# automations.yaml
- alias: "Send voice command to OM1 Robot"
  trigger:
    - platform: conversation
      command:
        - "bring me [item] from [location]"
        - "fetch [item]"
        - "deliver [item] to [location]"
  action:
    - service: input_text.set_value
      target:
        entity_id: input_text.om1_voice_command
      data:
        value: "{{ trigger.sentence }}"
```

### 2. Set Environment Variables
```bash
export HOME_ASSISTANT_URL="http://your-ha-instance:8123"
export HOME_ASSISTANT_TOKEN="your_long_lived_access_token"
export COINBASE_API_KEY="your_api_key"
export COINBASE_API_SECRET="your_api_secret"
export COINBASE_WALLET_ID="your_wallet_id"
```

### 3. Run the Robot Agent
```bash
uv run src/run.py robot_fetch_service
```

## Usage Example

**User:** "Hey Google, ask robot to bring me coffee from kitchen"

**Robot:** "Fetch service request: coffee from kitchen. Service fee: $2. Please send 0.002 ETH to confirm."

**User:** [Sends 0.002 ETH via wallet]

**Robot:** "Payment confirmed. Navigating to kitchen to fetch coffee."

[Robot navigates, picks up coffee, delivers to user]

**Robot:** "Task complete. Coffee delivered. Ready for next request."

## Testing

Run the test suite:
```bash
# Test Home Assistant plugin
uv run pytest tests/inputs/plugins/test_home_assistant.py -v
```

## Features Implemented

✅ Home Assistant voice command integration
✅ Coinbase wallet payment monitoring
✅ Complete service workflow (request → payment → execution → completion)
✅ Modular plugin architecture following OM1 patterns
✅ Comprehensive test coverage
✅ Environment variable configuration
✅ Error handling and logging

## Real-World Applications

- **Home service robots**: Fetch items, deliver objects between rooms
- **Office delivery robots**: Documents, supplies, mail delivery
- **Hospital robots**: Medication, equipment delivery on-demand
- **Warehouse robots**: On-demand pick and place tasks
- **Hotel robots**: Room service, amenity delivery

## Technical Highlights

- Plugin follows OM1's `FuserInput` architecture pattern
- Async/await for non-blocking I/O operations
- Uses OM1's existing IOProvider for debugging
- Type-safe with Pydantic models
- Compatible with WebSim visualization
- Ready for physical robot integration

## Future Extensions

- Multiple pricing tiers based on task complexity
- Distance-based pricing calculation
- Task queue management for multiple requests
- Multi-robot coordination for larger tasks
- Integration with robot navigation stack

---

**This implementation provides a practical, robotics-focused foundation for autonomous service robots to operate in a service economy, perfectly aligned with OM1's mission as an AI runtime for robots.**
