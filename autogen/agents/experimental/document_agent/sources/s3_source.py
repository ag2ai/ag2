# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any

from .....import_utils import optional_import_block, require_optional_import
from ..core.base_interfaces import DocumentSource
from ..core.config import SourceConfig

with optional_import_block():
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


@require_optional_import(["boto3"], "s3_source")
class S3DocumentSource(DocumentSource):
    """S3-compatible document source (works with AWS S3, MinIO, etc.)."""

    def __init__(self, config: SourceConfig):
        self.config = config
        self._client = None

        # Validate required config
        if not config.bucket_name:
            raise ValueError("bucket_name is required for S3 source")

        logger.info(f"S3DocumentSource initialized for bucket: {config.bucket_name}")

    @property
    def client(self) -> Any:
        """Lazy initialization of S3 client."""
        if self._client is None:
            try:
                client_kwargs = {"service_name": "s3", "region_name": self.config.region or "us-east-1"}

                # Add endpoint URL for MinIO or other S3-compatible services
                if self.config.endpoint_url:
                    client_kwargs["endpoint_url"] = self.config.endpoint_url

                # Add credentials if provided
                if self.config.access_key and self.config.secret_key:
                    client_kwargs["aws_access_key_id"] = self.config.access_key
                    client_kwargs["aws_secret_access_key"] = self.config.secret_key

                client = boto3.client(**client_kwargs)

                # Test connection
                client.head_bucket(Bucket=self.config.bucket_name)
                logger.info(f"Successfully connected to S3 bucket: {self.config.bucket_name}")

                self._client = client

            except NoCredentialsError:
                raise ValueError(
                    "AWS credentials not found. Provide access_key/secret_key or configure AWS credentials."
                )
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    raise ValueError(f"S3 bucket not found: {self.config.bucket_name}")
                elif error_code == "403":
                    raise ValueError(f"Access denied to S3 bucket: {self.config.bucket_name}")
                else:
                    raise ValueError(f"Failed to connect to S3: {e}")
            except ImportError:
                raise ImportError("boto3 is required for S3 document source. Install with: pip install boto3")

        return self._client

    def list_documents(self, prefix: str | None = None) -> list[str]:
        """List documents in S3 bucket."""
        try:
            full_prefix = ""
            if self.config.prefix:
                full_prefix = self.config.prefix.rstrip("/")
            if prefix:
                full_prefix = f"{full_prefix}/{prefix.lstrip('/')}" if full_prefix else prefix.lstrip("/")

            # Add trailing slash to treat as directory
            if full_prefix and not full_prefix.endswith("/"):
                full_prefix += "/"

            response = self.client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=full_prefix,
                MaxKeys=1000,  # Reasonable limit
            )

            documents = []
            for obj in response.get("Contents", []):
                key = obj["Key"]

                # Skip directories (keys ending with /)
                if key.endswith("/"):
                    continue

                # Remove the base prefix to get relative path
                if self.config.prefix:
                    base_prefix = self.config.prefix.rstrip("/") + "/"
                    if key.startswith(base_prefix):
                        key = key[len(base_prefix) :]

                # Filter by supported file types
                supported_extensions = [
                    "pdf",
                    "docx",
                    "pptx",
                    "xlsx",
                    "html",
                    "md",
                    "txt",
                    "json",
                    "csv",
                    "xml",
                    "adoc",
                    "png",
                    "jpg",
                    "jpeg",
                    "tiff",
                ]
                ext = Path(key).suffix.lower().lstrip(".")
                if ext in supported_extensions:
                    documents.append(key)

            logger.debug(f"Found {len(documents)} documents in S3 with prefix '{full_prefix}'")
            return documents

        except Exception as e:
            logger.error(f"Failed to list S3 documents: {e}")
            return []

    def download_document(self, document_key: str, local_path: Path) -> bool:
        """Download document from S3 to local path."""
        try:
            # Construct full S3 key
            s3_key = document_key
            if self.config.prefix:
                s3_key = f"{self.config.prefix.rstrip('/')}/{document_key}"

            # Create local directory
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            self.client.download_file(self.config.bucket_name, s3_key, str(local_path))

            logger.debug(f"Downloaded S3 object {s3_key} to {local_path}")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.error(f"S3 object not found: {s3_key}")
            else:
                logger.error(f"Failed to download {document_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to download {document_key}: {e}")
            return False

    def get_document_metadata(self, document_key: str) -> dict[str, Any]:
        """Get S3 object metadata."""
        try:
            s3_key = document_key
            if self.config.prefix:
                s3_key = f"{self.config.prefix.rstrip('/')}/{document_key}"

            response = self.client.head_object(Bucket=self.config.bucket_name, Key=s3_key)

            return {
                "size": response.get("ContentLength", 0),
                "modified_time": response.get("LastModified"),
                "content_type": response.get("ContentType"),
                "etag": response.get("ETag", "").strip('"'),
                "storage_class": response.get("StorageClass", "STANDARD"),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.debug(f"S3 object not found for metadata: {s3_key}")
            else:
                logger.error(f"Failed to get metadata for {document_key}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to get metadata for {document_key}: {e}")
            return {}

    def document_exists(self, document_key: str) -> bool:
        """Check if document exists in S3."""
        try:
            metadata = self.get_document_metadata(document_key)
            return len(metadata) > 0
        except Exception:
            return False
