from datetime import datetime
from typing import Optional

DATE_FORMAT = "%Y/%m/%d"
ISO_FORMAT = "%Y/%m/%d %H:%M:%S"


def get_datetime(date_format: Optional[str] = None) -> str:
    if not date_format:
        return datetime.now()
    return datetime.now().strftime(date_format)


def convert_date(
    date_str: str, input_format: str, output_format: Optional[str] = None
) -> str:
    if not output_format:
        return datetime.strptime(date_str, input_format)
    return datetime.strptime(date_str, input_format).strftime(output_format)


def parse_datetime(date_str: str, input_format: str = ISO_FORMAT) -> datetime:
    return datetime.strptime(date_str, input_format)
