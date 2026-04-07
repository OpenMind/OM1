from dataclasses import dataclass, field
from actions.base import Interface


@dataclass
class ScheduleCronJobInput:
    """Input interface for the ScheduleCronJob action."""

    schedule_time: str = field(
        metadata={"description": "Date and time when the job should first execute, formatted as 'YYYY-MM-DD HH:MM:SS'."}
    )

    function: str = field(
        metadata={"description": "The name of the function or action to call. Must exactly match a function name available in the tools list."}
    )

    args: str = field(
        default="{}",
        metadata={
            "description": (
                "JSON-encoded parameters for the function named in 'function'."
                " Look up that function's schema in the available tools and serialize its parameters as a JSON string."
                " Example: if function='speak' (which takes action: str), set args='{\"action\": \"Hello world\"}'."
                " Do NOT include the function name itself inside args."
            )
        },
    )

    recurrence: str = field(
        default="",
        metadata={
            "description": (
                "How often to repeat. Leave empty or 'once' for one-time tasks."
                " Supported: 'hourly', 'daily', 'weekly', 'every 30s', 'every 5m', 'every 2h', 'every 3d'."
            )
        },
    )


@dataclass
class ScheduleCronJob(Interface[ScheduleCronJobInput, ScheduleCronJobInput]):
    """Register a scheduled cron job to be executed at a specific date and time, optionally on a recurring schedule. Use this for any user request involving future or repeated tasks: one-time reminders (set schedule_time to the desired moment, leave recurrence empty) or recurring tasks (set schedule_time to the *first* occurrence and recurrence to the repeat pattern, e.g. 'daily', 'weekly', 'every 30m'). Always use the current date/time as context. Format schedule_time as 'YYYY-MM-DD HH:MM:SS'. Immediate Requests: If the user asks for something NOW (no future time or recurrence), call the appropriate tool directly — do NOT use schedule_cron_job. Scheduled Requests: If the user mentions a future time or recurrence, ALWAYS use schedule_cron_job. When filling schedule_cron_job: set 'function' to the user's original request with all time and schedule information stripped out (e.g. 'check the weather in NYC every 5 minutes' → function = 'check the weather in NYC'); leave 'args' as '{}' — do NOT extract function names or sub-arguments; set 'schedule_time' to the extracted date-time; set 'recurrence' to the extracted repeat pattern (e.g. 'every 5m'), or leave empty for one-time tasks."""

    # Signals that generate_function_schemas_from_actions should append the
    # list of other available action/tool names to this schema's description.
    inject_sub_function_names = True

    input: ScheduleCronJobInput
    output: ScheduleCronJobInput
