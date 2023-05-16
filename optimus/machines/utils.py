"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import datetime


def get_next_sunday(
    input_date: datetime.datetime,
    first_shift_hour: int = 0,
    first_shift_minute: int = 0,
    first_shift_second: int = 0,
):
    """
    Returns next Sunday for given input_date
    First shift tells at what time Sunday starts. By default at 00:00:00
    If input_date is Sunday itself, then return the same date

    Params
    -------------------------
    first_shift_hour: sets the hour of first shift
    first_shift_minute: sets the minute of first shift
    first_shift_second: sets the second of first shift
    """
    shifted_input_date = input_date - datetime.timedelta(
        hours=first_shift_hour,
        minutes=first_shift_minute,
        seconds=first_shift_second,
    )
    days_until_sunday = 6 - shifted_input_date.weekday()
    next_sunday = shifted_input_date + datetime.timedelta(days=days_until_sunday)
    next_sunday = datetime.datetime(
        year=next_sunday.year,
        month=next_sunday.month,
        day=next_sunday.day,
        hour=first_shift_hour,
        minute=first_shift_minute,
        second=first_shift_second,
    )

    return next_sunday
