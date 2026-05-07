# Petoi Bittle BLE Protocol Reference

Derived from source analysis of PetoiControllerQt.

---

## 1. Transport Stack

```
┌─────────────────────────────────────┐
│     Application (ASCII commands)    │  UiSerialHandler / UiMotionControl
├─────────────────────────────────────┤
│       QSerialPort abstraction       │  QSerialMessageQueue
├─────────────────────────────────────┤
│   Virtual serial device (/dev/...)  │  rfcomm or BLE UART shim
├─────────────────────────────────────┤
│   Nordic UART Service (NUS) / BLE   │  GATT over BLE
└─────────────────────────────────────┘
```

The device (`D0:EF:76:CD:BA:D6`, advertised as `BittleA6_SSP`) connects over
**Bluetooth LE** using the **Nordic UART Service**. BlueZ exposes this as a
virtual serial port that `QSerialPort` opens normally.

### NUS GATT characteristics

| Characteristic | UUID | Direction | D-Bus path |
|---|---|---|---|
| TX (device → host) | `6e400003-b5a3-f393-e0a9-e50e24dcca9e` | Notify | `.../service0028/char0029` |
| RX (host → device) | `6e400002-b5a3-f393-e0a9-e50e24dcca9e` | Write | `.../service0028/char002c` |

The RX characteristic requires enabling the Client Characteristic Configuration
Descriptor (CCCD, `0x2902`) on the TX characteristic before notifications flow.

### Default serial parameters

From `SerialConnectionPreference` defaults (index into the sorted combo lists):

| Parameter | Value |
|---|---|
| Baud rate | 115200 (index 7) |
| Data bits | 8 (index 3) |
| Stop bits | 1 (index 0, `OneStop`) |
| Parity | None (index 0) |

---

## 2. Wire Protocol: Petoi Token Protocol (ASCII)

All traffic in both directions is **plain ASCII text**. Commands are sent as
null-terminated strings written directly to the serial device; the firmware
replies with human-readable ASCII strings (plain text, whitespace-separated
values for multi-value responses).

There is no framing delimiter on the outgoing side — the app sends the string
bytes as-is (`msg.c_str()`, `msg.length()`). The firmware implicitly treats a
newline or a new token as a message boundary.

> **Note:** The source also contains a binary `SerialDataPacket` structure
> (`DataPacket.h`, `DataPacketHandler.cpp`) with a version byte, length field,
> two instruction bytes, error byte, type byte, and a 26-byte payload capped
> with `\r\n`. This format is fully defined but **not wired into outgoing
> traffic in this application**. All actual `sendCmdViaSerialPort` calls pass
> plain ASCII strings, and incoming data is also parsed as plain text. The
> binary protocol appears to be a design artifact or reserved for future use.

---

## 3. Command Reference

### 3.1 Motion / Gait commands

All motion commands use the token prefix `k` followed by a skill name.
The app polls key state every **100 ms** and re-sends the current gait command
whenever the state changes.

| ASCII command | Action |
|---|---|
| `kwkF` | Walk forward |
| `kwkL` | Walk left |
| `kwkR` | Walk right |
| `kbk` | Walk / crawl / run backward (shared) |
| `kcrF` | Crawl forward |
| `kcrL` | Crawl left |
| `kcrR` | Crawl right |
| `ktrF` | Trot (fast run) forward |
| `ktrL` | Trot left |
| `ktrR` | Trot right |
| `kbalance` | Stand still / balance (sent on key release) |

**Keyboard mapping** (keys held simultaneously select gait mode):

| Key | Direction |
|---|---|
| `W` | Forward |
| `S` | Backward |
| `A` | Left |
| `D` | Right |
| `1` | Normal walk mode |
| `2` | Trot/run mode |
| `3` | Crawl mode |

### 3.2 Posture / skill commands

One-shot commands triggered by UI buttons. No parameters.

| ASCII command | Posture / skill |
|---|---|
| `kbuttUp` | Butt up |
| `kck` | Check around (look around) |
| `kstr` | Stretch |
| `khi` | Greeting / wave |
| `kpee` | Pee pose |
| `kpu` | Push up |
| `krest` | Rest (lie down) |
| `kstp` | Stepping in place |
| `kbf` | Back flip |
| `ksit` | Sit down |
| `kbdF` | Bunny jump |
| `kvt` | Stepping / vibrate |

### 3.3 Calibration commands

Calibration is a stateful session initiated with `c` and committed with `s`.

| ASCII command | Effect |
|---|---|
| `c` | Enter calibration mode. Device responds with the 16 current offsets. |
| `c<N> <deg>` | Adjust servo `N` by `deg` degrees (range −9 to +9, integer). Example: `c8 -3` |
| `s` | Save calibration offsets to EEPROM and exit calibration mode. |

**Servo numbering used for calibration:**

| Servo index | Joint |
|---|---|
| 0 | Head |
| 8 | Front-left upper leg |
| 9 | Front-right upper leg |
| 10 | Rear-left upper leg |
| 11 | Rear-right upper leg |
| 12 | Front-left lower leg |
| 13 | Front-right lower leg |
| 14 | Rear-left lower leg |
| 15 | Rear-right lower leg |

Servos 1–7 are defined in firmware but not exposed in this controller UI.

**Calibration response format** (device → host, ASCII):

```
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
<val0>, <val1>, <val2>, ... <val15>,
```

The app extracts the 16 comma-separated values and maps them back to servos
0, 8–15 (indices 1–4 and 5–8 in the UI dropdown respectively).

### 3.4 Terminal / custom commands

The UI exposes a raw terminal input and a custom-action table (persisted in
`actions.json` as `[{"name": "...", "cmd": "..."}]`). Any string entered there
is sent verbatim, allowing direct use of any firmware token not covered above.

---

## 4. Binary DataPacket structure (defined, not used for TX)

Documented here for completeness in case the firmware sends structured
responses in this format.

```
Offset  Size  Field
0       1     version  — always 0x01
1       1     dataLen  — total packet size (header 8 bytes + payload + 2 bytes \r\n)
2       1     ins0     — command category (see below)
3       1     ins1     — sub-command
4       1     error    — error code
5       1     type     — payload data type
6..N    ≤24   rest     — payload bytes
N+0     1     0x0D (\r)
N+1     1     0x0A (\n)
```

Maximum packet size: **32 bytes**.

**ins0 categories:**

| Value | Constant | Meaning |
|---|---|---|
| `0x00` | `CMD_PETOI` | Unsolicited device signal |
| `0x01` | `CMD_DC` | Device control command |
| `0x02` | `CMD_MOTION` | Motion command |
| `0x03` | `CMD_SERVO` | Raw servo command |

**DC sub-commands (`CMD_DC`):**

| ins1 | Constant | Effect |
|---|---|---|
| `0xF0` | `DC_PETOI_HELLO` | Establish connection |
| `0xFF` | `DC_PETOI_BYE` | Disconnect |
| `0x00/01/02/03` | `DC_LEDS_*` | LED off / on / flash on / flash off |
| `0x0C` | `DC_LEDS_STATUS` | Query LED state |
| `0x10/11` | `DC_GYRO_OFF/ON` | Gyro control |
| `0x20/21` | `DC_SERVOS_OFF/ON` | All servos off / on |
| `0x30/31` | `DC_SPEAKER_RPT/PLY` | Upload / play melody |
| `0x40/41/4F` | `DC_BUZZER_*` | Buzzer stop / once / continuous |

**Data type constants:**

| Value | Type |
|---|---|
| `0x11` | unsigned char |
| `0x12` | char |
| `0x13` | unsigned int |
| `0x14` | int |
| `0x15` | unsigned long |
| `0x16` | long |
| `0x17` | float |
| `0x18` | double |
| `0x00` | none |

---

## 5. Source file map

| File | Role |
|---|---|
| `src/DataPacket/Definitions.h` | All protocol constants (opcodes, status codes) |
| `src/DataPacket/DataPacket.h` | `SerialDataPacket` struct definition |
| `src/DataPacket/DataPacketHandler.cpp` | Binary packet assembler / validator |
| `src/Serial/QSerialMessageQueue.cpp` | QSerialPort wrapper with receive queue |
| `src/Serial/RawMessage.cpp` | Heap-allocated byte buffer for queued messages |
| `src/Main/Components/Serials/UiSerialHandler.cpp` | Connect / send / receive + 10 ms poll timer |
| `src/Main/Components/DefaultControls/UiMotionControl.cpp` | Key→gait state machine, 100 ms dispatch timer |
| `src/Main/Components/Calibration/UiCalibrationCheck.cpp` | Calibration session + servo angle tracking |
| `src/Main/Components/Calibration/CalibFeedbackProcedure.cpp` | Regex parser for calibration response text |
| `src/Main/Components/CustomCmds/UiCustomActions.cpp` | User-defined command table |
| `src/Main/Components/CustomCmds/JsonParementer.cpp` | Persist custom commands to `actions.json` |
| `src/Main/MainWindow_apx.cpp` | Qt signal/slot wiring + button handlers |
| `src/Config/GlobalConfig.h` | File name constants (`preference.json`, `actions.json`) |
