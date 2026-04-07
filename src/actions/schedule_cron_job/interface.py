from dataclasses import dataclass, field
from actions.base import Interface

# ---------------------------------------------------------------------------
# Runtime-configurable docstring variants for ScheduleCronJob.
# configure_docstring() patches ScheduleCronJob.__doc__ based on the
# use_program_input setting so the LLM receives correct instructions.
# ---------------------------------------------------------------------------

_DOCSTRING_PROGRAM_INPUT = (
    "Register a scheduled cron job to be executed at a specific date and time, optionally on a recurring schedule."
    " Use this for any user request involving future or repeated tasks:"
    " one-time reminders (set schedule_time to the desired moment, leave recurrence empty) or"
    " recurring tasks (set schedule_time to the *first* occurrence and recurrence to the repeat pattern,"
    " e.g. 'daily', 'weekly', 'every 30m')."
    " Always use the current date/time as context. Format schedule_time as 'YYYY-MM-DD HH:MM:SS'."
    " Immediate Requests: If the user asks for something NOW (no future time or recurrence),"
    " call the appropriate tool directly — do NOT use schedule_cron_job."
    " Scheduled Requests: If the user mentions a future time or recurrence, ALWAYS use schedule_cron_job."
    " When filling schedule_cron_job:"
    " set 'function' to the user's original request with all time and schedule information stripped out"
    " (e.g. 'check the weather in NYC every 5 minutes' → function = 'check the weather in NYC');"
    " leave 'args' as '{}' — do NOT extract function names or sub-arguments;"
    " set 'schedule_time' to the extracted date-time;"
    " set 'recurrence' to the extracted repeat pattern (e.g. 'every 5m'), or leave empty for one-time tasks."
)

_DOCSTRING_ORCHESTRATORS = (
    "You are an intelligent task coordinator with access to Agent Actions and MCP Tools,"
    " plus a meta-tool called schedule_cron_job."
    " Always use the current date/time as context. Format schedule_time as 'YYYY-MM-DD HH:MM:SS'."
    " Immediate Requests: If the user asks for something NOW, call the specific tool directly — do NOT use schedule_cron_job."
    " Scheduled Requests: If the user mentions a future time or recurrence, ALWAYS use schedule_cron_job."
    " When wrapping in schedule_cron_job, handle Agent Actions and MCP Tools differently:"
    " For Agent Actions: set 'function' to the action name (e.g. 'speak', 'move') and"
    " set 'args' to a valid JSON string with the action's parameters matching its schema."
    " For MCP Tools: set 'function' to the MCP tool name and"
    " set 'args' to a JSON string {\"command\": \"<original user request with time/schedule info stripped>\"};"
    " e.g. 'tell me the weather in NYC every 10 minutes' → function = <mcp-tool-name>,"
    " args = {\"command\": \"tell me the weather in NYC\"}."
    " Set 'schedule_time' to the extracted date-time and 'recurrence' to the repeat pattern"
    " (e.g. 'every 5m', 'daily'), or leave recurrence empty for one-time tasks."
)


def configure_docstring(use_program_input: bool) -> None:
    """Patch ``ScheduleCronJob.__doc__`` based on the ``use_program_input`` setting.

    Must be called before the LLM is loaded so that both text-based and
    function-calling LLMs see the correct description.
    """
    ScheduleCronJob.__doc__ = (
        _DOCSTRING_PROGRAM_INPUT if use_program_input else _DOCSTRING_ORCHESTRATORS
    )


@dataclass
class ScheduleCronJobInput:
    """Input interface for the ScheduleCronJob action."""

    schedule_time: str
    """Date-time string for when the job should first execute (e.g. '2025-01-30 15:04:00')."""

    function: str
    """use_program_input=True: the original user request with time info stripped.
    use_program_input=False + Agent Action: the action name to call.
    use_program_input=False + MCP Tool: the MCP tool name."""

    args: str = field(default="{}")
    """use_program_input=True: leave as '{}'.
    use_program_input=False + Agent Action: JSON string of action parameters.
    use_program_input=False + MCP Tool: JSON string {"command": "<stripped user request>"}."""

    recurrence: str = field(default="")
    """How often to repeat.  Leave empty or use 'once' for a one-time task.
    Supported patterns:
      - '' or 'once'   — run once at schedule_time
      - 'hourly'       — repeat every 60 minutes
      - 'daily'        — repeat every 24 hours
      - 'weekly'       — repeat every 7 days
      - 'every Xs'     — repeat every X seconds  (e.g. 'every 30s')
      - 'every Xm'     — repeat every X minutes  (e.g. 'every 30m')
      - 'every Xh'     — repeat every X hours    (e.g. 'every 2h')
      - 'every Xd'     — repeat every X days     (e.g. 'every 3d')
    """


@dataclass
class ScheduleCronJob(Interface[ScheduleCronJobInput, ScheduleCronJobInput]):
    """Register a cron job to be executed at a specific date and time, optionally on a recurring schedule.

    Use this for any user request involving future or repeated tasks:
      - One-time reminders: set schedule_time to the desired moment, leave recurrence empty.
      - Recurring tasks: set schedule_time to the *first* occurrence and recurrence to the
        desired repeat pattern (e.g. 'daily', 'weekly', 'every 30m').

    Always use the current date/time as context. Format schedule_time as 'YYYY-MM-DD HH:MM:SS'.

    You are an intelligent task coordinator. You have access to a set of specialized tools (e.g., get_weather, send_email) and a meta-tool called schedule_cron_job.

    Your Operating Logic:

    Immediate Requests: If the user asks for something NOW (e.g., 'What's the weather?'), call the specific tool directly (e.g., get_weather).

    Scheduled Requests: If the user mentions a time, duration, or recurrence (e.g., 'Every 20s', 'Tomorrow at 5pm'), you MUST NOT call the functional tool directly. Instead, call schedule_cron_job.

    Wrapping: Inside schedule_cron_job, fill function with the name of the tool the user wants, and map the user's details into args based on that tool's specific JSON Schema.

    Recurrence Standard: Always format recurrence as 'every [number][s/m/h/d]' for recurring tasks.
    """

    # Signals that generate_function_schemas_from_actions should append the
    # list of other available action/tool names to this schema's description.
    inject_sub_function_names = True

    input: ScheduleCronJobInput
    output: ScheduleCronJobInput
