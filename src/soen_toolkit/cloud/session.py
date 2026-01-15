"""AWS session management with pre-flight validation.

Validates all AWS resources (credentials, bucket, role, ECR image) before
any job submission to fail fast and provide clear error messages.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError

if TYPE_CHECKING:
    from mypy_boto3_ecr import ECRClient
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_sagemaker import SageMakerClient
    from mypy_boto3_sts import STSClient

    from .config import AWSConfig, CloudConfig

logger = logging.getLogger(__name__)


class CloudSessionError(Exception):
    """Base exception for cloud session errors."""

    pass


class CredentialsError(CloudSessionError):
    """AWS credentials are missing or invalid."""

    pass


class BucketAccessError(CloudSessionError):
    """S3 bucket does not exist or is not accessible."""

    pass


class RoleValidationError(CloudSessionError):
    """IAM role validation failed."""

    pass


class ImageValidationError(CloudSessionError):
    """ECR image validation failed."""

    pass


@dataclass
class AWSIdentity:
    """Represents the authenticated AWS identity."""

    account_id: str
    user_arn: str
    user_id: str


class CloudSession:
    """AWS session with pre-flight validation.

    All validation is performed in __init__ to fail fast before any
    resources are provisioned.

    Attributes:
        config: The cloud configuration.
        boto_session: The underlying boto3 session.
        identity: The authenticated AWS identity.
    """

    def __init__(
        self,
        config: CloudConfig | AWSConfig,
        *,
        skip_validation: bool = False,
    ) -> None:
        """Initialize cloud session with validation.

        Args:
            config: Cloud or AWS configuration.
            skip_validation: If True, skip pre-flight checks (for testing).

        Raises:
            CredentialsError: If AWS credentials are missing or invalid.
            BucketAccessError: If S3 bucket is not accessible.
            RoleValidationError: If IAM role is invalid.
        """
        # Extract AWS config if CloudConfig provided
        from .config import CloudConfig

        self._cloud_config: CloudConfig | None
        if isinstance(config, CloudConfig):
            self._cloud_config = config
            self._aws_config = config.aws
        else:
            self._cloud_config = None
            self._aws_config = config

        # Create boto session with retry configuration
        self._boto_session = self._create_boto_session()

        # Run pre-flight validation
        if not skip_validation:
            self._identity = self._validate_credentials()
            self._validate_bucket()
            self._validate_role()

    @property
    def config(self) -> AWSConfig:
        """Get AWS configuration."""
        return self._aws_config

    @property
    def cloud_config(self) -> CloudConfig | None:
        """Get full cloud configuration if available."""
        return self._cloud_config

    @property
    def boto_session(self) -> boto3.Session:
        """Get underlying boto3 session."""
        return self._boto_session

    @property
    def identity(self) -> AWSIdentity:
        """Get authenticated AWS identity."""
        return self._identity

    @property
    def region(self) -> str:
        """Get AWS region."""
        return self._aws_config.region

    @property
    def bucket(self) -> str:
        """Get S3 bucket name."""
        return self._aws_config.bucket

    def _create_boto_session(self) -> boto3.Session:
        """Create boto3 session with retry configuration."""
        return boto3.Session(region_name=self._aws_config.region)

    def _get_boto_config(self) -> BotoConfig:
        """Get boto client configuration with retry settings."""
        return BotoConfig(
            retries={
                "mode": "adaptive",
                "max_attempts": 5,
            },
            connect_timeout=10,
            read_timeout=60,
        )

    def s3_client(self) -> S3Client:
        """Get S3 client."""
        return self._boto_session.client("s3", config=self._get_boto_config())

    def sts_client(self) -> STSClient:
        """Get STS client."""
        return self._boto_session.client("sts", config=self._get_boto_config())

    def sagemaker_client(self) -> SageMakerClient:
        """Get SageMaker client."""
        return self._boto_session.client("sagemaker", config=self._get_boto_config())

    def ecr_client(self, region: str | None = None) -> ECRClient:
        """Get ECR client for specified region."""
        return self._boto_session.client(
            "ecr",
            region_name=region or self._aws_config.region,
            config=self._get_boto_config(),
        )

    def _validate_credentials(self) -> AWSIdentity:
        """Validate AWS credentials by calling STS GetCallerIdentity.

        Returns:
            AWSIdentity with account and user information.

        Raises:
            CredentialsError: If credentials are missing or invalid.
        """
        logger.info("Validating AWS credentials...")
        try:
            sts = self.sts_client()
            response = sts.get_caller_identity()
            identity = AWSIdentity(
                account_id=response["Account"],
                user_arn=response["Arn"],
                user_id=response["UserId"],
            )
            logger.info(f"Authenticated as: {identity.user_arn}")
            return identity
        except NoCredentialsError as e:
            raise CredentialsError(
                "AWS credentials not found. Configure credentials via:\n"
                "  - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
                "  - AWS credentials file (~/.aws/credentials)\n"
                "  - IAM role (if running on EC2/ECS)"
            ) from e
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            raise CredentialsError(
                f"AWS credential validation failed ({error_code}): {e}"
            ) from e

    def _validate_bucket(self) -> None:
        """Validate S3 bucket exists and is accessible.

        Raises:
            BucketAccessError: If bucket doesn't exist or isn't accessible.
        """
        bucket = self._aws_config.bucket
        logger.info(f"Validating S3 bucket: {bucket}")
        try:
            s3 = self.s3_client()
            s3.head_bucket(Bucket=bucket)
            logger.info(f"Bucket verified: {bucket}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404":
                raise BucketAccessError(
                    f"S3 bucket '{bucket}' does not exist. "
                    f"Create it with: aws s3 mb s3://{bucket}"
                ) from e
            elif error_code == "403":
                raise BucketAccessError(
                    f"Access denied to S3 bucket '{bucket}'. "
                    f"Check IAM permissions for s3:ListBucket and s3:GetObject."
                ) from e
            else:
                raise BucketAccessError(
                    f"S3 bucket validation failed ({error_code}): {e}"
                ) from e

    def _validate_role(self) -> None:
        """Validate IAM role exists (best-effort check).

        Note: We can't fully validate the role without assuming it,
        so we just check it exists via IAM.GetRole if we have permission.

        Raises:
            RoleValidationError: If role validation fails.
        """
        role_arn = self._aws_config.role
        logger.info(f"Validating IAM role: {role_arn}")

        # Extract role name from ARN
        # Format: arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME
        try:
            role_name = role_arn.split("/")[-1]
        except (IndexError, AttributeError) as e:
            raise RoleValidationError(f"Invalid role ARN format: {role_arn}") from e

        # Best-effort: try to describe the role
        # This requires iam:GetRole permission which we may not have
        try:
            iam = self._boto_session.client("iam", config=self._get_boto_config())
            iam.get_role(RoleName=role_name)
            logger.info(f"Role verified: {role_name}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchEntity":
                raise RoleValidationError(
                    f"IAM role '{role_name}' does not exist. "
                    f"Create a SageMaker execution role with appropriate permissions."
                ) from e
            elif error_code == "AccessDenied":
                # We don't have permission to check, but that's OK
                logger.warning(
                    f"Cannot verify role '{role_name}' (no iam:GetRole permission). "
                    f"Assuming role exists."
                )
            else:
                logger.warning(f"Role validation warning ({error_code}): {e}")

    def validate_ecr_image(self, image_uri: str) -> bool:
        """Validate ECR image exists and is accessible.

        Args:
            image_uri: ECR image URI in format:
                ACCOUNT.dkr.ecr.REGION.amazonaws.com/REPO:TAG

        Returns:
            True if image is valid and accessible.

        Raises:
            ImageValidationError: If image validation fails.
        """
        if not image_uri:
            raise ImageValidationError("image_uri cannot be empty")

        # Check if it's an ECR URI
        if ".dkr.ecr." not in image_uri:
            logger.info(f"Non-ECR image URI, skipping validation: {image_uri}")
            return True

        logger.info(f"Validating ECR image: {image_uri}")

        try:
            # Parse ECR URI: ACCOUNT.dkr.ecr.REGION.amazonaws.com/REPO:TAG
            parts = image_uri.split("/")
            domain = parts[0]
            repo_tag = "/".join(parts[1:])

            if ":" in repo_tag:
                repo, tag = repo_tag.rsplit(":", 1)
            else:
                repo = repo_tag
                tag = "latest"

            # Extract region from domain
            region = domain.split(".")[3]

            ecr = self.ecr_client(region=region)
            ecr.describe_images(
                repositoryName=repo, imageIds=[{"imageTag": tag}]
            )
            logger.info(f"ECR image verified: {image_uri}")
            return True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "ImageNotFoundException":
                raise ImageValidationError(
                    f"ECR image not found: {image_uri}. "
                    f"Build and push the image first."
                ) from e
            elif error_code == "RepositoryNotFoundException":
                raise ImageValidationError(
                    f"ECR repository not found for: {image_uri}. "
                    f"Create the repository first."
                ) from e
            else:
                raise ImageValidationError(
                    f"ECR image validation failed ({error_code}): {e}"
                ) from e
        except (IndexError, ValueError) as e:
            raise ImageValidationError(
                f"Invalid ECR image URI format: {image_uri}"
            ) from e

    def upload_file(self, local_path: str, s3_key: str) -> str:
        """Upload a local file to S3.

        Args:
            local_path: Path to local file.
            s3_key: S3 object key (without bucket).

        Returns:
            Full S3 URI of uploaded file.

        Raises:
            FileNotFoundError: If local file doesn't exist.
            ClientError: If upload fails.
        """
        from pathlib import Path

        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        s3 = self.s3_client()
        bucket = self._aws_config.bucket

        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        s3.upload_file(str(path), bucket, s3_key)

        return f"s3://{bucket}/{s3_key}"

    def upload_directory(self, local_dir: str, s3_prefix: str) -> int:
        """Upload a local directory to S3.

        Args:
            local_dir: Path to local directory.
            s3_prefix: S3 prefix for uploaded files.

        Returns:
            Number of files uploaded.

        Raises:
            FileNotFoundError: If local directory doesn't exist.
            ClientError: If upload fails.
        """
        import os
        from pathlib import Path

        dir_path = Path(local_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        s3 = self.s3_client()
        bucket = self._aws_config.bucket
        count = 0

        for root, _, files in os.walk(dir_path):
            for filename in files:
                local_path = Path(root) / filename
                rel_path = local_path.relative_to(dir_path).as_posix()
                s3_key = f"{s3_prefix}/{rel_path}"

                logger.debug(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
                s3.upload_file(str(local_path), bucket, s3_key)
                count += 1

        logger.info(f"Uploaded {count} files to s3://{bucket}/{s3_prefix}")
        return count

    # -------------------- S3 URI Parsing --------------------

    @staticmethod
    def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        """Parse an S3 URI into bucket and key.

        Args:
            s3_uri: Full S3 URI in format s3://bucket/key/path

        Returns:
            Tuple of (bucket, key).

        Raises:
            ValueError: If URI format is invalid.
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI (must start with s3://): {s3_uri}")

        without_scheme = s3_uri[5:]  # Remove 's3://'
        parts = without_scheme.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        if not bucket:
            raise ValueError(f"Invalid S3 URI (missing bucket): {s3_uri}")

        return bucket, key

    # -------------------- Upload to Any Bucket --------------------

    def upload_file_to_uri(self, local_path: str, s3_uri: str) -> str:
        """Upload a local file to any S3 bucket.

        Args:
            local_path: Path to local file.
            s3_uri: Full S3 URI (s3://bucket/key).

        Returns:
            The S3 URI of uploaded file.

        Raises:
            FileNotFoundError: If local file doesn't exist.
            ValueError: If S3 URI is invalid.
            ClientError: If upload fails.
        """
        from pathlib import Path

        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        bucket, key = self.parse_s3_uri(s3_uri)
        s3 = self.s3_client()

        logger.info(f"Uploading {local_path} to {s3_uri}")
        s3.upload_file(str(path), bucket, key)

        return s3_uri

    def upload_directory_to_uri(self, local_dir: str, s3_uri: str) -> int:
        """Upload a local directory to any S3 bucket.

        Args:
            local_dir: Path to local directory.
            s3_uri: Full S3 URI prefix (s3://bucket/prefix).

        Returns:
            Number of files uploaded.

        Raises:
            FileNotFoundError: If local directory doesn't exist.
            ValueError: If S3 URI is invalid.
            ClientError: If upload fails.
        """
        import os
        from pathlib import Path

        dir_path = Path(local_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        bucket, prefix = self.parse_s3_uri(s3_uri)
        # Remove trailing slash from prefix for consistent joining
        prefix = prefix.rstrip("/")

        s3 = self.s3_client()
        count = 0

        for root, _, files in os.walk(dir_path):
            for filename in files:
                local_path = Path(root) / filename
                rel_path = local_path.relative_to(dir_path).as_posix()
                s3_key = f"{prefix}/{rel_path}" if prefix else rel_path

                logger.debug(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
                s3.upload_file(str(local_path), bucket, s3_key)
                count += 1

        logger.info(f"Uploaded {count} files to s3://{bucket}/{prefix}")
        return count

    # -------------------- Download from Any Bucket --------------------

    def download_file_from_uri(self, s3_uri: str, local_path: str) -> str:
        """Download a file from any S3 bucket.

        Args:
            s3_uri: Full S3 URI (s3://bucket/key).
            local_path: Local file path to save to.

        Returns:
            The local path where file was saved.

        Raises:
            ValueError: If S3 URI is invalid.
            ClientError: If download fails (e.g., file not found).
        """
        from pathlib import Path

        bucket, key = self.parse_s3_uri(s3_uri)

        if not key:
            raise ValueError(f"S3 URI must include a key/path: {s3_uri}")

        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)

        s3 = self.s3_client()

        logger.info(f"Downloading {s3_uri} to {local_path}")
        s3.download_file(bucket, key, str(local))

        return str(local)

    def download_directory_from_uri(self, s3_uri: str, local_dir: str) -> int:
        """Download all files under an S3 prefix to a local directory.

        Args:
            s3_uri: Full S3 URI prefix (s3://bucket/prefix).
            local_dir: Local directory to save files to.

        Returns:
            Number of files downloaded.

        Raises:
            ValueError: If S3 URI is invalid.
            ClientError: If download fails.
        """
        from pathlib import Path

        bucket, prefix = self.parse_s3_uri(s3_uri)
        # Remove trailing slash for consistent handling
        prefix = prefix.rstrip("/")

        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)

        s3 = self.s3_client()
        count = 0

        # List all objects under the prefix
        paginator = s3.get_paginator("list_objects_v2")
        # Add trailing slash to prefix to ensure we get contents, not prefix matches
        list_prefix = f"{prefix}/" if prefix else ""

        for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]

                # Compute relative path from the prefix
                if prefix:
                    rel_path = s3_key[len(prefix) :].lstrip("/")
                else:
                    rel_path = s3_key

                if not rel_path:
                    # Skip the prefix itself if it's listed as an object
                    continue

                local_path = local / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                logger.debug(f"Downloading s3://{bucket}/{s3_key} to {local_path}")
                s3.download_file(bucket, s3_key, str(local_path))
                count += 1

        logger.info(f"Downloaded {count} files from s3://{bucket}/{prefix} to {local_dir}")
        return count

    def is_s3_directory(self, s3_uri: str) -> bool:
        """Check if an S3 URI points to a 'directory' (prefix with objects under it).

        Args:
            s3_uri: Full S3 URI to check.

        Returns:
            True if the URI is a directory (has objects underneath), False if it's a single file.
        """
        bucket, key = self.parse_s3_uri(s3_uri)
        s3 = self.s3_client()

        # First, check if the exact key exists as a file
        try:
            s3.head_object(Bucket=bucket, Key=key)
            # If this succeeds, it's a file
            return False
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code != "404":
                raise

        # If exact key doesn't exist, check if there are objects with this prefix
        prefix = key.rstrip("/") + "/" if key else ""
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)

        return response.get("KeyCount", 0) > 0

