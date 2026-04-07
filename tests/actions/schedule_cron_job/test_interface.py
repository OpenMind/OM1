from actions.schedule_cron_job.interface import ScheduleCronJob, ScheduleCronJobInput


class TestScheduleCronJobInput:
    def test_required_fields(self):
        inp = ScheduleCronJobInput(schedule_time="2026-01-01 10:00:00", function="speak")
        assert inp.schedule_time == "2026-01-01 10:00:00"
        assert inp.function == "speak"

    def test_default_args(self):
        inp = ScheduleCronJobInput(schedule_time="2026-01-01 10:00:00", function="speak")
        assert inp.args == "{}"

    def test_default_recurrence(self):
        inp = ScheduleCronJobInput(schedule_time="2026-01-01 10:00:00", function="speak")
        assert inp.recurrence == ""

    def test_custom_args(self):
        inp = ScheduleCronJobInput(
            schedule_time="2026-01-01 10:00:00",
            function="speak",
            args='{"action": "Hello"}',
        )
        assert inp.args == '{"action": "Hello"}'

    def test_custom_recurrence(self):
        inp = ScheduleCronJobInput(
            schedule_time="2026-01-01 10:00:00",
            function="speak",
            recurrence="daily",
        )
        assert inp.recurrence == "daily"


class TestScheduleCronJob:
    def test_inject_sub_function_names(self):
        assert ScheduleCronJob.inject_sub_function_names is True

    def test_creation(self):
        inp = ScheduleCronJobInput(schedule_time="2026-01-01 10:00:00", function="speak")
        job = ScheduleCronJob(input=inp, output=inp)
        assert job.input is inp
        assert job.output is inp
