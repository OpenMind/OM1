---
title: Transition Rules
description: "Introduction to transition rules"
---

Transition rules define how and when the robot switches between operational modes. Each rule specifies a source mode, a target mode, a trigger mechanism, and an execution priority. The system continuously evaluates these rules and executes the most appropriate transition based on current inputs, elapsed time, and contextual conditions.

Transition rules are defined as an array of rule objects under the transition_rules configuration.

## Rule Schema

Each transition rule is represented as an object with required and optional fields.

### Required Fields

| Field                  | Type            | Required | Description                                                                                                                              |
| ---------------------- | --------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `from_mode`            | `string`        | Yes      | The operational mode in which this transition rule is evaluated                                                                          |
| `to_mode`              | `string`        | Yes      | The target mode to transition into when the rule is triggered                                                                            |
| `transition_type`      | `string`        | Yes      | Defines how the transition is triggered. Supported values: `input_triggered`, `time_based`, `context_aware`                              |
| `priority`             | `integer`       | Yes      | Determines rule precedence when multiple transition rules are eligible (higher value takes precedence)                                   |
| `trigger_keywords`     | `array<string>` | No       | List of keywords or phrases that can trigger the transition via user input; applicable to `input_triggered` and `time_based` transitions |
| `cooldown_seconds`     | `number`        | No       | Minimum time in seconds before the same transition rule can be triggered again, preventing rapid or repeated transitions                 |
| `context_conditions`   | `object`        | No       | Contextual conditions that must be satisfied for a `context_aware` transition to be eligible                                             |

Supported Operators in `context_conditions`

| Operator   | Value Type          | Description                                                               |
| ---------- | ------------------- | ------------------------------------------------------------------------- |
| `min`      | `number`            | Minimum allowed numeric value                                             |
| `max`      | `number`            | Maximum allowed numeric value                                             |
| `contains` | `string` or `array` | Checks whether a string contains a substring or an array contains a value |
| `one_of`   | `array`             | Evaluates to true if the context value matches any value in the array     |
| `not`      | `any`               | Negates the specified condition                                           |

### Priority Resolution

When multiple transition rules are eligible:

- Rules are sorted by priority (higher values take precedence)

- The highest-priority rule is selected
