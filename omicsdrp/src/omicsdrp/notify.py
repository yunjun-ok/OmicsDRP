"""Email notifications for the pipeline (best-effort, never fatal).

Sends via ``msmtp`` (an authenticated Gmail SMTP relay configured in
``~/.msmtprc`` -> app password in ``~/.omicsdrp_smtp_pass``). Gmail rejects
unauthenticated direct sends, so a plain local MTA/mailx cannot deliver; msmtp
authenticates and delivers. Falls back to ``mailx`` only if msmtp is absent.
Failures are logged and never interrupt training.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from email.message import EmailMessage
from typing import List, Optional

DEFAULT_SENDER = os.environ.get("EMAIL_FROM", "jmj3078@gmail.com")


def resolve_recipient(to: Optional[str]) -> Optional[str]:
    return to or os.environ.get("EMAIL_TO") or None


def email_available() -> bool:
    return shutil.which("msmtp") is not None or shutil.which("mailx") is not None


def _send_msmtp(subject: str, body: str, to: str, sender: str,
                attachments: Optional[List[str]]) -> None:
    msg = EmailMessage()
    msg["From"] = f"OmicsDRP <{sender}>"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    for a in attachments or []:
        if os.path.isfile(a):
            with open(a, "rb") as f:
                msg.add_attachment(f.read(), maintype="text", subtype="markdown",
                                   filename=os.path.basename(a))
    subprocess.run(["msmtp", "-f", sender, to], input=msg.as_bytes(),
                   check=True, timeout=60,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _send_mailx(subject: str, body: str, to: str,
                attachments: Optional[List[str]]) -> None:
    cmd = ["mailx", "-s", subject]
    for a in attachments or []:
        if os.path.isfile(a):
            cmd += ["-A", a]
    cmd += [to]
    subprocess.run(cmd, input=body.encode(), timeout=30, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def send_email(subject: str, body: str, to: Optional[str],
               attachments: Optional[List[str]] = None,
               sender: str = DEFAULT_SENDER) -> bool:
    """Send one mail. Returns True on success, False otherwise. No-ops (returns
    False) when no recipient is configured or no transport exists."""
    to = resolve_recipient(to)
    if not to:
        return False
    try:
        if shutil.which("msmtp"):
            _send_msmtp(subject, body, to, sender, attachments)
        elif shutil.which("mailx"):
            _send_mailx(subject, body, to, attachments)
        else:
            print("[email] neither msmtp nor mailx on PATH; skipping")
            return False
        print(f"[email] sent '{subject}' -> {to}")
        return True
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode(errors="replace").strip()
        print(f"[email] failed ('{subject}' -> {to}): rc={e.returncode} {err}")
        return False
    except Exception as e:  # never let a mail failure break the sweep
        print(f"[email] failed ('{subject}' -> {to}): {e}")
        return False
