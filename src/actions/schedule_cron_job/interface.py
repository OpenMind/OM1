from dataclasses import dataclass, field

from actions.base import Interface


@dataclass
class ScheduleCronJobInput:
    """
    Input interface for the ScheduleCronJob action.

    Parameters
    ----------
    schedule_time : str
        Date and time when the job should first execute, formatted as 'YYYY-MM-DD HH:MM:SS'.
    function : str
        The user's original request with all time and schedule information stripped out.
        For example: 'check the weather in NYC every 5 minutes' → 'check the weather in NYC'.
    recurrence : str
        How often to repeat. Leave empty or use 'once' for one-time tasks.
        Supported values: 'hourly', 'daily', 'weekly', 'every Xs', 'every Xm', 'every Xh', 'every Xd'
        (e.g. 'every 30s', 'every 5m', 'every 2h', 'every 3d').
        Do NOT use cron expressions (e.g. '* * * * *') — they are not supported.
    """

    schedule_time: str
    function: str
    recurrence: str = field(default="")


@dataclass
class ScheduleCronJob(Interface[ScheduleCronJobInput, ScheduleCronJobInput]):
    """
    Register a scheduled cron job to execute at a specific time, optionally on a recurring schedule.

    Effect: Schedules the given function/request to run at schedule_time with an
    optional recurrence pattern. Use this for any user request involving a future
    or repeated task — one-time reminders or recurring jobs.

    Use this action when the user mentions a future time or recurrence pattern.
    Do NOT use this action for immediate requests; call the appropriate tool directly instead.

    When filling schedule_time: format as 'YYYY-MM-DD HH:MM:SS' using the current date/time as context.
    When filling function: strip all time and schedule information from the user's request
    (e.g. 'check the weather in NYC every 5 minutes' → 'check the weather in NYC').
    When filling recurrence: use one of the supported plain-English patterns — 'hourly', 'daily',
    'weekly', or 'every Xm' / 'every Xh' / 'every Xd' (e.g. 'every 5m', 'every 2h', 'every 3d').
    Leave empty for one-time tasks. Do NOT use cron expressions (e.g. '* * * * *').
    """

    input: ScheduleCronJobInput
    output: ScheduleCronJobInput
