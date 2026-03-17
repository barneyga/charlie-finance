"""Email notification for critical (red) alerts. Uses only stdlib."""
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone

from charlie.config import Settings
from charlie.storage.db import Database

logger = logging.getLogger(__name__)


def send_alert_email(settings: Settings, db: Database, alerts: list[dict]) -> bool:
    """Send an email for red-level alerts. Returns True on success.

    Silently returns False if SMTP is not configured or if there are
    no red alerts. Marks sent alerts as notified in the DB.
    """
    red_alerts = [a for a in alerts if a.get("level") == "red"]
    if not red_alerts:
        return False

    if not settings.smtp_host or not settings.alert_email_to:
        logger.info("SMTP not configured — skipping email for %d red alerts", len(red_alerts))
        return False

    # Build email body
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject = f"Charlie Finance Alert — {len(red_alerts)} critical signal{'s' if len(red_alerts) > 1 else ''}"

    rows = []
    for a in red_alerts:
        rows.append(
            f"<tr><td style='padding:6px;border:1px solid #ddd'>{a['name']}</td>"
            f"<td style='padding:6px;border:1px solid #ddd;text-align:center'>"
            f"<span style='background:#dc3545;color:#fff;padding:2px 8px;border-radius:4px'>"
            f"{a['level'].upper()}</span></td>"
            f"<td style='padding:6px;border:1px solid #ddd'>{a['value']:.2f}</td>"
            f"<td style='padding:6px;border:1px solid #ddd'>{a['message']}</td></tr>"
        )

    html = f"""\
<html>
<body style="font-family:Arial,sans-serif;color:#333">
<h2 style="color:#dc3545">Charlie Finance — Critical Alert</h2>
<p>Generated at {now}</p>
<table style="border-collapse:collapse;width:100%">
<tr style="background:#f5f5f5">
  <th style="padding:8px;border:1px solid #ddd;text-align:left">Metric</th>
  <th style="padding:8px;border:1px solid #ddd">Level</th>
  <th style="padding:8px;border:1px solid #ddd">Value</th>
  <th style="padding:8px;border:1px solid #ddd;text-align:left">Details</th>
</tr>
{"".join(rows)}
</table>
<p style="color:#888;font-size:12px;margin-top:20px">
  Sent by Charlie Finance alerting engine. Configure thresholds in config/alerts.yaml.
</p>
</body>
</html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.smtp_user or "charlie-finance@localhost"
    msg["To"] = settings.alert_email_to
    msg.attach(MIMEText(html, "html"))

    try:
        if settings.smtp_port == 465:
            server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=30)
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)
            server.starttls()

        if settings.smtp_user and settings.smtp_password:
            server.login(settings.smtp_user, settings.smtp_password)

        server.sendmail(msg["From"], [settings.alert_email_to], msg.as_string())
        server.quit()
        logger.info("Alert email sent to %s (%d alerts)", settings.alert_email_to, len(red_alerts))

        # Mark alerts as notified
        for a in red_alerts:
            if "id" in a:
                db.conn.execute(
                    "UPDATE alerts SET notified = 1 WHERE id = ?", (a["id"],)
                )
        db.conn.commit()
        return True

    except Exception as exc:
        logger.error("Failed to send alert email: %s", exc)
        return False
