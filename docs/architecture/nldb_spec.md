\# Natural Language Data Bus (NLDB) – Draft Specification



\*\*Status:\*\* Draft / Proposal  

\*\*Audience:\*\* OM1 core contributors, robotics developers, and plugin authors



The Natural Language Data Bus (NLDB) is a central abstraction in OM1.  

Instead of shipping large, heterogeneous sensor payloads across the system, OM1 normalizes

perception into \*natural language events\* that downstream components (LLMs, planners, tools,

and external systems) can consume.



This document proposes a first-pass technical specification for NLDB messages:

\- a common \*\*message schema\*\*

\- \*\*required vs optional\*\* fields

\- \*\*validation rules\*\*

\- \*\*guidelines for plugin authors\*\* who want to publish new inputs into NLDB



The goal is to turn the existing architectural idea…



> sensors → VLM / ASR / other encoders → \*\*NLDB\*\* → state fuser → LLM planners → actions



…into something that can be implemented consistently across agents and hardware platforms.



---



\## 1. Design Goals



NLDB messages are designed to be:



1\. \*\*Language-centric\*\*  

&nbsp;  All payloads must contain a human-readable `natural\_language\_description` field that

&nbsp;  summarizes “what the robot currently perceives or believes”.



2\. \*\*Modality-aware\*\*  

&nbsp;  Messages include enough metadata (`modality`, `source`, `position\_estimate`, etc.)

&nbsp;  for downstream modules to understand \*where\* the information came from.



3\. \*\*Extensible but constrained\*\*  

&nbsp;  A small set of \*\*required\*\* fields gives us consistency, while `metadata` and

&nbsp;  `tags` allow experiment-specific extensions without breaking consumers.



4\. \*\*Implementation-agnostic\*\*  

&nbsp;  This spec uses JSON examples, but NLDB messages can be transported via ROS2, Zenoh,

&nbsp;  CycloneDDS, websockets, or any other middleware used by OM1 agents.



---



\## 2. Base Message Schema



Unless otherwise noted, NLDB messages are assumed to be UTF-8 JSON objects

published on a bus topic/channel (e.g. `nldb.raw`, `nldb.fused`) by encoders,

sensors, or intermediate components.



\### 2.1. Top-level fields



| Field                          | Type              | Required | Description |

| ------------------------------ | ----------------- | -------- | ----------- |

| `id`                           | string            | ✅       | Globally unique message ID (e.g. UUIDv4). |

| `schema\_version`               | string            | ✅       | Version tag for this spec, e.g. `"1.0.0"`. |

| `timestamp`                    | string (ISO 8601) | ✅       | Event time in UTC, e.g. `"2025-03-01T12:00:32.123Z"`. |

| `source`                       | string            | ✅       | Logical name of the producer, e.g. `"camera\_front"`, `"mic\_0"`, `"web\_event"`. |

| `modality`                     | string (enum)     | ✅       | Primary modality (see below). |

| `natural\_language\_description` | string            | ✅       | Human-readable description of the event. |

| `language`                     | string            | ⚪️      | BCP-47 language tag, default `"en"`. |

| `confidence`                   | number \[0.0–1.0]  | ⚪️      | Model confidence in the description. |

| `position\_estimate`            | object / array    | ⚪️      | Estimated position of the referenced object(s) or viewpoint. |

| `raw\_data\_ref`                 | string (URI)      | ⚪️      | Reference/handle to original raw data (image, audio, point cloud, etc.). |

| `agent\_context`                | object            | ⚪️      | IDs linking this message to the current agent/session/task. |

| `tags`                         | string\[]          | ⚪️      | Short labels (e.g. `\["human", "object.apple", "table"]`). |

| `metadata`                     | object            | ⚪️      | Transport- and model-specific metadata (see below). |



\### 2.2. Modality values



Suggested canonical values:



\- `"vision"` – camera images, depth, video frames  

\- `"audio"` – microphone streams, ASR results  

\- `"text"` – direct text input (chat, web page, commands)  

\- `"lidar"` – LIDAR / depth sensors  

\- `"state"` – fused state summaries  

\- `"event"` – discrete events (button press, system event, tool callback)  

\- `"proprioception"` – joint states, IMU, robot internal sensors  

\- `"other"` – anything that doesn’t fit yet (should be avoided in production; prefer adding new values)



Downstream components \*\*must not\*\* rely on free-form modality strings; any new modality

should be documented and added to this list in a follow-up patch.



\### 2.3. Agent context



`agent\_context` is an optional nested object that binds an NLDB message to

higher-level execution context:



```json

"agent\_context": {

&nbsp; "agent\_id": "spot\_v1",

&nbsp; "session\_id": "sess\_2025\_03\_01\_001",

&nbsp; "task\_id": "tid\_pickup\_apple",

&nbsp; "episode\_id": "episode\_7"

}

```



None of these fields are required individually, but including \*at least one\*

stable identifier is strongly recommended whenever the message is part of

a long-running interaction.



\### 2.4. Position estimate



This is deliberately flexible, but should be documented per agent. Two common patterns:



\*\*Simple 3D position\*\*



```json

"position\_estimate": {

&nbsp; "frame": "map",

&nbsp; "x": 1.2,

&nbsp; "y": 0.3,

&nbsp; "z": 0.8

}

```



\*\*Array form\*\*



```json

"position\_estimate": \[1.2, 0.3, 0.8]

```



In both cases, coordinate frame semantics (`map`, `odom`, `base\_link`, etc.) must

be documented in the agent configuration.



\### 2.5. Metadata



`metadata` is a free-form JSON object reserved for implementation details

that should not be promoted to top-level fields yet. Typical values:



```json

"metadata": {

&nbsp; "encoder\_model": "gpt-4o-mini",

&nbsp; "encoder\_type": "vlm",

&nbsp; "latency\_ms": 32,

&nbsp; "frame\_index": 123,

&nbsp; "sensor\_fov\_deg": 90

}

```



Consumers must treat unknown `metadata` keys as optional hints only.



---



\## 3. Validation Rules



To guarantee a minimum level of consistency across agents, the following rules apply:



1\. `id` must be a non-empty string and \*\*unique\*\* at least within a session.  

2\. `schema\_version` must be a non-empty string; for this document we assume `"1.0.0"`.  

3\. `timestamp` \*\*must\*\* be a valid RFC3339 / ISO 8601 string in UTC (`...Z`).  

4\. `source` must be a non-empty string and stable per logical sensor/encoder.  

5\. `modality` must be one of the allowed values listed in §2.2.  

6\. `natural\_language\_description` must be a non-empty string.  

7\. If present, `confidence` must lie in the closed interval `\[0.0, 1.0]`.  

8\. If present, `tags` must be an array of non-empty strings.  

9\. Large binary payloads (images, audio, point clouds) \*\*must not\*\* be embedded directly

&nbsp;  in the NLDB message; they should be referenced via `raw\_data\_ref`.



A JSON Schema (or Pydantic model) for this spec can be added in a follow-up PR.



---



\## 4. Example Messages



\### 4.1. Vision caption from front camera



```json

{

&nbsp; "id": "7a7c8e3e-8d7b-4e0c-a3a3-f3f8b8c8c001",

&nbsp; "schema\_version": "1.0.0",

&nbsp; "timestamp": "2025-03-01T12:00:32.123Z",

&nbsp; "source": "camera\_front",

&nbsp; "modality": "vision",

&nbsp; "natural\_language\_description": "You see a red apple on a wooden table in front of you.",

&nbsp; "language": "en",

&nbsp; "confidence": 0.92,

&nbsp; "position\_estimate": {

&nbsp;   "frame": "map",

&nbsp;   "x": 1.2,

&nbsp;   "y": 0.3,

&nbsp;   "z": 0.8

&nbsp; },

&nbsp; "raw\_data\_ref": "blob://camera\_front/frame\_00123",

&nbsp; "agent\_context": {

&nbsp;   "agent\_id": "spot\_v1",

&nbsp;   "session\_id": "sess\_2025\_03\_01\_001"

&nbsp; },

&nbsp; "tags": \["object.apple", "table", "indoor"],

&nbsp; "metadata": {

&nbsp;   "encoder\_model": "gpt-4o-mini",

&nbsp;   "encoder\_type": "vlm",

&nbsp;   "latency\_ms": 28

&nbsp; }

}

```



\### 4.2. Audio → ASR transcript



```json

{

&nbsp; "id": "e8fd6d2e-f145-41cb-8a4f-90d4b5330f02",

&nbsp; "schema\_version": "1.0.0",

&nbsp; "timestamp": "2025-03-01T12:01:05.501Z",

&nbsp; "source": "mic\_headset",

&nbsp; "modality": "audio",

&nbsp; "natural\_language\_description": "Can you bring me the red apple from the table?",

&nbsp; "language": "en",

&nbsp; "confidence": 0.96,

&nbsp; "raw\_data\_ref": "blob://mic\_headset/segment\_00042",

&nbsp; "agent\_context": {

&nbsp;   "agent\_id": "spot\_v1",

&nbsp;   "session\_id": "sess\_2025\_03\_01\_001",

&nbsp;   "task\_id": "tid\_pickup\_apple"

&nbsp; },

&nbsp; "tags": \["human\_speech", "command"],

&nbsp; "metadata": {

&nbsp;   "encoder\_model": "whisper-large-v3",

&nbsp;   "encoder\_type": "asr",

&nbsp;   "latency\_ms": 120

&nbsp; }

}

```



\### 4.3. Fused state summary (output of State Fuser)



```json

{

&nbsp; "id": "f3c58ca0-22b1-4e6c-8fd5-21d7bfa4f900",

&nbsp; "schema\_version": "1.0.0",

&nbsp; "timestamp": "2025-03-01T12:01:10.000Z",

&nbsp; "source": "state\_fuser",

&nbsp; "modality": "state",

&nbsp; "natural\_language\_description": "You are standing in the kitchen near a wooden table. There is a red apple on the table about one meter in front of you. A human is sitting on a chair nearby and just asked you to bring them the apple.",

&nbsp; "language": "en",

&nbsp; "agent\_context": {

&nbsp;   "agent\_id": "spot\_v1",

&nbsp;   "session\_id": "sess\_2025\_03\_01\_001",

&nbsp;   "episode\_id": "episode\_7"

&nbsp; },

&nbsp; "tags": \["summary", "task\_context"],

&nbsp; "metadata": {

&nbsp;   "inputs\_fused": \[

&nbsp;     "7a7c8e3e-8d7b-4e0c-a3a3-f3f8b8c8c001",

&nbsp;     "e8fd6d2e-f145-41cb-8a4f-90d4b5330f02"

&nbsp;   ]

&nbsp; }

}

```



---



\## 5. State Fuser Interface (Informal)



The \*\*State Fuser\*\* is a logical component that:



1\. Subscribes to “raw” NLDB topics (e.g. `nldb.raw.\*`).

2\. Maintains a rolling window / buffer of recent NLDB messages.

3\. Periodically emits a \*fused summary\* message with `modality: "state"`.



This document does \*\*not\*\* prescribe a specific algorithm for fusing state  

(LLM-based summarization, rule-based merging, etc.), but recommends:



\- Fused messages should be \*\*idempotent snapshots\*\* of the current situation,

&nbsp; not incremental patches.

\- Downstream planners should be able to depend on the existence of at least one

&nbsp; recent `modality: "state"` message when making decisions.



A future “State Fuser Design” document can standardize this interface in more detail.



---



\## 6. Plugin Author Guidelines



If you are writing a new encoder or sensor plugin that publishes to NLDB:



1\. \*\*Always emit valid base fields\*\*



&nbsp;  - `id`, `schema\_version`, `timestamp`, `source`, `modality`,

&nbsp;    and `natural\_language\_description` must be present.



2\. \*\*Prefer stable `source` names\*\*



&nbsp;  Use meaningful and stable identifiers like `"camera\_front"`, `"mic\_room"`,

&nbsp;  `"webhook\_shopify"`, rather than `"camera\_1"` if possible.



3\. \*\*Avoid large payloads in NLDB\*\*



&nbsp;  Images, audio, and other heavy data should live in separate storage or

&nbsp;  streams, referenced with `raw\_data\_ref`.



4\. \*\*Document your modality-specific extensions\*\*



&nbsp;  If you need additional fields for a specific domain, put them in `metadata`

&nbsp;  and document them in the agent’s README or in a follow-up spec.



5\. \*\*Keep descriptions grounded\*\*



&nbsp;  `natural\_language\_description` should describe \*what is actually observed\*,

&nbsp;  not high-level plans or hypothetical futures (those belong to planner outputs).



---



\## 7. Versioning and Forward Compatibility



\- The `schema\_version` field allows multiple versions of this spec to coexist.

\- New \*\*optional\*\* fields can be added in minor versions (e.g. `1.1.0`) without

&nbsp; breaking existing consumers.

\- Removing or changing the meaning of \*\*required\*\* fields requires a major version bump

&nbsp; (e.g. `2.0.0`) and must be coordinated with OM1 core maintainers.



For now, this document proposes `schema\_version: "1.0.0"` as a starting point for

experimentation. Feedback from real deployments should drive future revisions.



---



\## 8. Open Questions / Future Work



\- Should NLDB support \*\*bidirectional\*\* messages (e.g. “downlink” decisions) in the

&nbsp; same schema, or should those live on a separate bus?

\- Do we want a canonical \*\*JSON Schema\*\* or \*\*Pydantic model\*\* in the OM1 codebase?

\- How should NLDB interact with \*\*on-chain state\*\* in FABRIC for verifiable logging?



These topics are intentionally left open so that early adopters can experiment.

This document is meant to be a concrete starting point for discussion and iteration.



