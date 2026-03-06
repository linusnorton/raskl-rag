"""SES email notifications for versioned JSONL diffs."""

from __future__ import annotations

import html
import logging
import os

import boto3

from ras_docproc.diff import DiffReport

logger = logging.getLogger(__name__)

NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", "linusnorton@gmail.com")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "linusnorton@gmail.com")


def _format_html(report: DiffReport) -> str:
    """Format a DiffReport as an HTML email body."""
    rows = [
        f"<tr><td>Added</td><td>{report.blocks_added}</td></tr>",
        f"<tr><td>Removed</td><td>{report.blocks_removed}</td></tr>",
        f"<tr><td>Changed</td><td>{report.blocks_changed}</td></tr>",
        f"<tr><td>Unchanged</td><td>{report.blocks_unchanged}</td></tr>",
    ]
    summary_table = f"<table border='1' cellpadding='4'><tr><th>Status</th><th>Count</th></tr>{''.join(rows)}</table>"

    diffs_html = ""
    if report.changed_blocks:
        diffs_html = "<h3>Changed Blocks</h3>"
        for bd in report.changed_blocks[:20]:  # Limit to 20 diffs in email
            diffs_html += f"<h4>{html.escape(bd.block_id)}</h4>"
            diffs_html += f"<pre>{html.escape(bd.unified_diff)}</pre>"
        if len(report.changed_blocks) > 20:
            diffs_html += f"<p>... and {len(report.changed_blocks) - 20} more changed blocks</p>"

    meta_html = ""
    if report.meta_diff:
        meta_html = f"<h3>Metadata Changes</h3><pre>{html.escape(report.meta_diff)}</pre>"

    return f"""\
<html><body>
<h2>{html.escape(report.doc_id)} — v{report.old_version} → v{report.new_version}</h2>
{summary_table}
{meta_html}
{diffs_html}
</body></html>"""


def send_diff_email(report: DiffReport, region: str = "eu-west-2") -> None:
    """Send a diff report email via SES."""
    if not report.has_changes:
        logger.info("No changes in %s v%d — skipping email", report.doc_id, report.new_version)
        return

    subject = (
        f"[raskl-rag] {report.doc_id} v{report.new_version}: "
        f"{report.blocks_added} added, {report.blocks_changed} changed, {report.blocks_removed} removed"
    )

    ses = boto3.client("ses", region_name=region)
    ses.send_email(
        Source=SENDER_EMAIL,
        Destination={"ToAddresses": [NOTIFY_EMAIL]},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {"Html": {"Data": _format_html(report), "Charset": "UTF-8"}},
        },
    )
    logger.info("Sent diff email for %s v%d to %s", report.doc_id, report.new_version, NOTIFY_EMAIL)
