"""Who is using the viewer: the name written next to the regions they draw.

Several people outline brains on one lab machine, often under one account, so
the account name is only the starting guess. The name typed in the Brain regions
panel is remembered and wins from then on.
"""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

from .utils import get_logger

logger = get_logger("user")


def _file() -> Path:
    from .runtime import app_data_dir

    return app_data_dir() / "user.json"


def account_name() -> str:
    """The logged-in account's full name if the system has one, else its login."""
    try:
        import pwd  # not on Windows

        full = pwd.getpwuid(os.getuid()).pw_gecos.split(",")[0].strip()
        if full:
            return full
    except (ImportError, KeyError, AttributeError):
        pass
    try:
        return getpass.getuser()
    except Exception:
        return ""


def user_name() -> str:
    """The name to credit: the one typed last, or the account's."""
    try:
        name = str(json.loads(_file().read_text(encoding="utf-8")).get("name", "")).strip()
        if name:
            return name
    except (OSError, ValueError, AttributeError):
        pass
    return account_name()


def set_user_name(name: str) -> None:
    """Remember *name* for next time. Blank goes back to the account's name."""
    try:
        target = _file()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"name": str(name).strip()}), encoding="utf-8")
    except OSError:
        logger.debug("could not remember the user name", exc_info=True)


#: Attribute stored with each region outline: who drew (last saved) it.
DRAWN_BY = "drawn_by"


def credit() -> dict[str, str]:
    """Attributes to store with an outline so the workbook can say who drew it."""
    name = user_name()
    return {DRAWN_BY: name} if name else {}
