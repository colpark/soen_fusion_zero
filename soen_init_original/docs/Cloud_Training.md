---
layout: default
title: Cloud Training
---

# Cloud Training

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Unit_Converter.md" style="margin-right: 2em;">&#8592; Previous: Unit Converter</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Docker_Setup.md" style="margin-left: 2em;">Next: Docker Setup &#8594;</a>
</div>

Train SOEN models on AWS SageMaker with GPU acceleration.

| Method | Best For |
|--------|----------|
| **Integrated** | Simplest - just add `cloud: active: true` to your training config |
| **Cloud GUI** | Visual job management, beginners |
| **CLI** | Automation, scripting, power users |

---

## How It Works

Before diving into setup, it helps to understand what's happening behind the scenes.

### The Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  YOUR MACHINE                                                           │
│  ┌─────────────────┐                                                    │
│  │ Training Config │──┐                                                 │
│  │(experiment.yaml)│  │                                                 │
│  └─────────────────┘  │                                                 │
│  ┌─────────────────┐  │    Upload configs/data                          │
│  │ Local Data      │──┼─────────────────────────┐                       │
│  │ (optional)      │  │                         │                       │
│  └─────────────────┘  │                         ▼                       │
└───────────────────────┼─────────────────────────────────────────────────┘
                        │                    ┌────────────────┐
                        │                    │  S3 Bucket     │
                        │                    │  (working      │
                        │                    │   storage)     │
                        │                    └───────┬────────┘
                        │                            │
                        │    Submit job              │ Read inputs
                        ▼                            ▼
              ┌─────────────────────────────────────────────────┐
              │  AWS SAGEMAKER                                  │
              │  ┌───────────────────────────────────────────┐  │
              │  │  GPU Instance (ml.g5.xlarge, etc.)        │  │
              │  │  ┌─────────────────────────────────────┐  │  │
              │  │  │  Docker Container                   │  │  │
              │  │  │  - SOEN Toolkit code                │  │  │
              │  │  │  - Your training config             │  │  │
              │  │  │  - Your data                        │  │  │
              │  │  └─────────────────────────────────────┘  │  │
              │  └───────────────────────────────────────────┘  │
              └──────────────────────┬──────────────────────────┘
                                     │
                    Logs metrics     │     Saves artifacts
                    & params         │     (checkpoints, models)
                         ▼           ▼
              ┌──────────────┐   ┌────────────────┐
              │ MLflow Server│   │ S3 Bucket      │
              │ (optional)   │   │ (output)       │
              └──────────────┘   └────────────────┘
```

### What Each Component Does

| Component | Purpose |
|-----------|---------|
| **S3 Bucket** | Temporary storage for configs, data, and outputs during training |
| **IAM Role** | Permissions that let SageMaker access S3, ECR, and run jobs |
| **Docker Image** | Container with SOEN code and dependencies |
| **MLflow Server** | Tracks experiments, stores metrics and artifacts (optional) |

### What Gets Uploaded to S3

When you submit a job, the system uploads to your S3 bucket:

```
s3://your-bucket/soen/project/experiment/job-name/timestamp/
├── config/
│   └── training_config.yaml    # Your training configuration
├── data/                       # Only if using local data
│   └── (your data files)
└── output/                     # Where results go
    └── (checkpoints, models)
```

Each job gets a unique path, so multiple jobs never conflict.

---

## Team Setup vs. Individual Setup

**This is important:** Many resources can be shared across a team. You don't all need separate everything.

### Shared Resources (Set Up Once Per Team)

| Resource | Why Shared | Who Sets It Up |
|----------|------------|----------------|
| **IAM Role** | Permissions are the same for everyone | Team admin |
| **Docker Images** | Same code for everyone | Team admin (rebuild when toolkit updates) |
| **MLflow Server** | Centralized experiment tracking | Team admin |
| **S3 Bucket** | Can share (jobs use unique paths) | Team admin or individual choice |

### Individual Resources (Each Team Member)

| Resource | Why Individual | What To Do |
|----------|----------------|------------|
| **AWS Credentials** | Each person needs their own login | Get IAM user from admin |
| **Environment Variables** | Points to shared/individual resources | Copy from team template |

---

## Joining an Existing Team

If your team already has cloud training set up, you only need to:

### 1. Get AWS Credentials

Ask your team admin to create an IAM user for you. They'll give you:
- Access Key ID
- Secret Access Key

Configure the AWS CLI:
```bash
aws configure
# AWS Access Key ID: (paste yours)
# AWS Secret Access Key: (paste yours)
# Default region: us-east-1
# Default output format: json
```

Verify it works:
```bash
aws sts get-caller-identity
# Should show your user ARN
```

### 2. Set Environment Variables

Get the values from your team (they'll look something like this):

```bash
# Add to ~/.bashrc or ~/.zshrc

# AWS / SageMaker (get these from your team)
export SOEN_SM_ROLE="arn:aws:iam::123456789012:role/sagemaker-role"
export SOEN_SM_BUCKET="team-soen-training"
export AWS_REGION="us-east-1"

# Docker images (shared across team)
export SOEN_DOCKER_PYTORCH="123456789012.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit:v2-pytorch"
export SOEN_DOCKER_JAX="123456789012.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit:v2-jax"

# MLflow (optional, if team uses it)
export MLFLOW_TRACKING_URI="https://mlflow.yourteam.com"
```

Reload your shell:
```bash
source ~/.zshrc  # or ~/.bashrc
```

### 3. Test It

```bash
# Verify bucket access
aws s3 ls s3://team-soen-training

# Try a cost estimate (doesn't actually run anything)
python -m soen_toolkit.cloud train --config your_config.yaml --estimate
```

**That's it.** You're ready to submit jobs.

---

## First-Time Setup (New Team)

If you're setting up cloud training from scratch:

### 1. Create an S3 Bucket

The bucket stores your training configs, data, and results temporarily.

```bash
# Bucket names must be globally unique across all of AWS
aws s3 mb s3://your-team-soen-training --region us-east-1
```

### 2. Create a SageMaker IAM Role

This role gives SageMaker permission to run jobs and access your bucket.

```bash
# Create the role
aws iam create-role \
    --role-name sagemaker-role \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'

# Attach required policies
aws iam attach-role-policy --role-name sagemaker-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy --role-name sagemaker-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Get the role ARN (you'll need this)
aws iam get-role --role-name sagemaker-role --query 'Role.Arn' --output text
# Example output: arn:aws:iam::123456789012:role/sagemaker-role
```

### 3. Build Docker Images

See [Docker Setup](Docker_Setup.md) for detailed instructions. Summary:

```bash
# Create ECR repository
aws ecr create-repository --repository-name soen-toolkit --region us-east-1

# Build and push (from project root)
./src/soen_toolkit/cloud/docker/build.sh \
    123456789012.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit \
    v2
```

### 4. Set Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export SOEN_SM_ROLE="arn:aws:iam::123456789012:role/sagemaker-role"
export SOEN_SM_BUCKET="your-team-soen-training"
export AWS_REGION="us-east-1"
export SOEN_DOCKER_PYTORCH="123456789012.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit:v2-pytorch"
export SOEN_DOCKER_JAX="123456789012.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit:v2-jax"
```

### 5. Create IAM Users for Team Members

For each team member:

```bash
# Create user
aws iam create-user --user-name colleague-name

# Attach policies
aws iam attach-user-policy --user-name colleague-name \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-user-policy --user-name colleague-name \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-user-policy --user-name colleague-name \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

# Create access keys (give these to the team member)
aws iam create-access-key --user-name colleague-name
```

Share the environment variable template with them (step 4) so they can copy it.

---

## Submitting Jobs

### Option A: Integrated (Simplest)

Add a `cloud:` section to your training config:

```yaml
# your_training_config.yaml
cloud:
  active: true
  # These can come from environment variables instead:
  # role: "arn:aws:iam::123456789012:role/sagemaker-role"
  # bucket: "team-soen-training"
  
  # Optional overrides:
  instance_type: ml.g5.xlarge
  use_spot: true
  max_runtime_hours: 24.0

training:
  # ... your normal training config
data:
  # ... your normal data config
model:
  # ... your normal model config
```

Then run normally:
```bash
train your_training_config.yaml
# or: python -m soen_toolkit.training your_training_config.yaml
```

The job is automatically submitted to the cloud:
```
============================================================
  Cloud job submitted: soen-training-20241127120000
============================================================
  Instance: ml.g5.xlarge
  Backend:  jax
  Region:   us-east-1

  Console:  https://us-east-1.console.aws.amazon.com/sagemaker/...

  Monitor with:
    python -m soen_toolkit.cloud status soen-training-...
    python -m soen_toolkit.cloud_gui
============================================================
```

**Environment variable fallbacks:**
- `SOEN_SM_ROLE` → `cloud.role`
- `SOEN_SM_BUCKET` → `cloud.bucket`
- `AWS_REGION` → `cloud.region`
- `SOEN_DOCKER_PYTORCH` → `cloud.docker_image_pytorch`
- `SOEN_DOCKER_JAX` → `cloud.docker_image_jax`

If you set environment variables, you can just use:

```yaml
cloud:
  active: true
```

### Option B: Cloud GUI

```bash
python -m soen_toolkit.cloud_gui
```

The GUI has tabs for:
- **Credentials** - Configure AWS settings (auto-loads from env vars)
- **Submit** - Select config, choose instance, submit job
- **Jobs** - Monitor running/completed jobs, view logs
- **Pricing** - Browse instance costs

Click **Save as Default** in Credentials to persist settings to `~/.soen/cloud_config.yaml`.

### Option C: CLI

```bash
# Basic (uses environment variables)
python -m soen_toolkit.cloud train --config experiment.yaml

# Show cost estimate only (doesn't submit)
python -m soen_toolkit.cloud train --config experiment.yaml --estimate

# Override settings
python -m soen_toolkit.cloud train \
    --config experiment.yaml \
    --instance-type ml.g5.2xlarge \
    --backend jax \
    --no-spot

# Monitor jobs
python -m soen_toolkit.cloud status <job-name>
python -m soen_toolkit.cloud list
python -m soen_toolkit.cloud stop <job-name>
```

---

## Cost Management

### Always Check Costs First

```bash
python -m soen_toolkit.cloud train --config experiment.yaml --estimate
```

### Instance Recommendations

| Use Case | Instance | On-Demand | Spot (~65% off) |
|----------|----------|-----------|-----------------|
| Quick tests | ml.g5.xlarge | $1.01/hr | $0.35/hr |
| Standard training | ml.g5.2xlarge | $1.52/hr | $0.53/hr |
| Large models | ml.g5.4xlarge | $2.53/hr | $0.89/hr |
| Multi-GPU | ml.g5.12xlarge | $7.09/hr | $2.48/hr |

### Spot vs. On-Demand

**Spot instances** are ~65% cheaper but can be interrupted if AWS needs capacity.

Use spot for:
- Experiments under 2 hours
- Jobs with checkpointing enabled

Use on-demand for:
- Long runs
- Production training

---

## MLflow Integration

If your team has an MLflow server, training metrics and artifacts are automatically tracked:

```yaml
# In training config
logging:
  mlflow_active: true
  mlflow_tracking_uri: https://mlflow.yourteam.com
  mlflow_password: "xxx"  # if auth required
```

MLflow uses its own S3 bucket for artifact storage (separate from the SageMaker bucket).

---

## Troubleshooting

### "AWS credentials not found"

```bash
# Check credentials
aws sts get-caller-identity

# If not configured
aws configure
```

### "Access denied to bucket"

Your IAM user needs S3 permissions. Ask your team admin to attach the `AmazonS3FullAccess` policy.

### "Waiting for capacity..."

Spot instances aren't available. Options:
1. Wait (AWS may find capacity)
2. Stop and resubmit without spot: `use_spot: false`
3. Try a different instance type

### Job fails immediately

Check CloudWatch logs in the GUI (Jobs tab → select job → view logs).

Common causes:
- Wrong Docker image
- Missing model/data files
- Config file errors

---

## Configuration Reference

### Cloud Config (cloud_config.yaml)

```yaml
aws:
  role: arn:aws:iam::123456789012:role/sagemaker-role
  bucket: team-soen-training
  region: us-east-1

instance:
  instance_type: ml.g5.xlarge
  instance_count: 1
  use_spot: true
  max_runtime_hours: 24.0

mlflow:
  tracking_uri: https://mlflow.yourteam.com
  experiment_name: my-experiment

project: my-project
experiment: experiment-001

docker_images:
  pytorch: 123456789012.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit:v2-pytorch
  jax: 123456789012.dkr.ecr.us-east-1.amazonaws.com/soen-toolkit:v2-jax
```

---

## Next Steps

- [Docker Setup](Docker_Setup.md) - Build and push Docker images
- [Training Models](Training_Models.md) - Training configuration reference
- [MLflow](MLFLOW.md) - Experiment tracking setup

<div align="center" style="margin-top: 2em;">
  <a href="Unit_Converter.md" style="margin-right: 2em;">&#8592; Previous: Unit Converter</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Docker_Setup.md" style="margin-left: 2em;">Next: Docker Setup &#8594;</a>
</div>
