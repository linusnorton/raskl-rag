"""Alibaba Function Compute handler: OSS event → download PDF → pipeline → versioned OSS output.

FC Custom Runtime receives events as HTTP POST to port 9000.
OSS events have a different format from AWS S3 events — this module translates.
Uses boto3 with OSS S3-compatible endpoint for file operations.
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

FC_PORT = int(os.environ.get("FC_CUSTOM_LISTEN_PORT", "9000"))


def _oss_event_to_s3_event(oss_event: dict) -> dict:
    """Convert FC OSS event format to AWS S3 event format for handler reuse."""
    records = []
    for event in oss_event.get("events", []):
        oss = event.get("oss", {})
        bucket = oss.get("bucket", {})
        obj = oss.get("object", {})
        records.append(
            {
                "s3": {
                    "bucket": {"name": bucket.get("name", "")},
                    "object": {"key": obj.get("key", "")},
                }
            }
        )
    return {"Records": records}


def _setup_oss_boto3():
    """Configure boto3 to use OSS S3-compatible endpoint."""
    import boto3
    from botocore.config import Config as BotoConfig

    region = os.environ.get("ALIBABA_REGION", "ap-southeast-1")
    endpoint = f"https://oss-{region}.aliyuncs.com"

    # OSS S3-compatible API requires signature v2
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("OSS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("OSS_ACCESS_KEY_SECRET", ""),
        config=BotoConfig(signature_version="s3"),
        region_name=region,
    )


class FCHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            oss_event = json.loads(body) if body else {}
            logger.info("FC event: %s", json.dumps(oss_event)[:500])

            # Patch boto3 S3 client to use OSS
            import ras_docproc.lambda_handler as handler

            handler.s3 = _setup_oss_boto3()

            # Convert OSS event to S3 format and delegate
            s3_event = _oss_event_to_s3_event(oss_event)
            result = handler.lambda_handler(s3_event, None)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            logger.exception("FC handler error")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        """Health check."""
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        logger.info(format, *args)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", FC_PORT), FCHandler)
    logger.info("FC docproc handler listening on port %d", FC_PORT)
    server.serve_forever()
