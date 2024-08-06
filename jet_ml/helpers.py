def hms_string(sec_elapsed):
    """
    Converts a time duration from seconds to a string formatted as h:m:s.

    Parameters:
    sec_elapsed (float): Time duration in seconds.

    Returns:
    str: Formatted string representing the time duration.
    """
    h = int(sec_elapsed // 3600)
    m = int((sec_elapsed % 3600) // 60)
    s = int(sec_elapsed % 60)
    return f"{h}:{m:02}:{s:02}"