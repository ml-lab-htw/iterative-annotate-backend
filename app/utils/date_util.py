def add_suffix_to_day(day):
    """
    Adds an ordinal suffix to a given day number.

    Args:
        day (int): The day of the month.

    Returns:
        str: The day of the month with an ordinal suffix.
    """
    if 4 <= day <= 20 or 24 <= day <= 30:
        return str(day) + "th"
    else:
        return str(day) + ["st", "nd", "rd"][day % 10 - 1]
   

def format_date(date):
    """
    Formats a datetime.date object into a human-readable string.

    Args:
        date (datetime.date): The date to format.

    Returns:
        str: The formatted date string.
    """
    day = add_suffix_to_day(date.day)
    month = date.strftime("%B")  # Full month name
    year = date.year
    return f"{day} {month} {year}"
