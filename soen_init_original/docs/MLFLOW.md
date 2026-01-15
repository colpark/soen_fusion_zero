---
layout: default
title: MLflow Server Setup for SOEN Toolkit
---
# MLflow Server Setup for SOEN Toolkit

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="index.md" class="nav-home">Home</a>
</div>

This directory contains scripts and documentation for setting up a shared MLflow tracking server on AWS.

## What This Provides

- **Automatic experiment tracking**: Training runs are logged to a shared team server by default
- **S3 artifact storage**: Checkpoints and `.soen` files automatically saved to S3
- **Rich web UI**: Browse, compare, and search experiments with plots, metrics, and artifacts
- **Team collaboration**: Everyone sees the same experiments without sharing AWS credentials
- **Simple setup**: Just add your password to the training config

## Quick Start

This will only have to be done once to set up the initial mlflow-server. Once up don't run this again.


### MLflow Web Interface

![MLflow Web Interface](Figures/Training_Models/MLFlow/screenshot_mlflow.png)

The MLflow web interface provides a rich dashboard for tracking and comparing your experiments. You can view metrics, parameters, artifacts, and add descriptions to your runs.


### For Team Members (Most Common)

**Just add your password to any training config:**

```yaml
logging:
  mlflow_active: true
  mlflow_password: "ask_team_for_password"
  # That's it! Everything else is automatic.
```

**Then train as normal** - your runs appear at: https://mlflow-greatsky.duckdns.org

### For Local Development (Local MLflow UI)

**Override to use local tracking:**

```yaml
logging:
  # 'file:./mlruns' is a URI that points to a local directory named './mlruns'.
  # MLflow will create this folder if it does not exist.
  mlflow_tracking_uri: "file:./mlruns"
  # View locally with:
  #   mlflow ui --backend-store-uri ./mlruns
  # Then open http://127.0.0.1:5000

If you prefer an absolute path:

```yaml
logging:
  mlflow_active: true
  mlflow_tracking_uri: "file:/absolute/path/to/your/project/mlruns"
```

Both forms are equivalent; the important bit is the 'file:' scheme, which tells MLflow
to use a directory on your filesystem as the tracking store. 
```

### To Disable MLflow

```yaml
logging:
  mlflow_active: false
```
Or omit from yaml.

## Server Setup (One-Time, Already Done)

The team server is already running. If you need to set up a new server:

```bash
chmod +x setup-mlflow-server.sh
./setup-mlflow-server.sh  # Creates all AWS infrastructure
```

### Access the Web UI

Open **https://mlflow-greatsky.duckdns.org** and login with:
- Username: `admin` 
- Password: (the one you added to config)

**Technical details**: The server runs MLflow natively on EC2 (t3.small) behind an Nginx reverse proxy with HTTP Basic Authentication. We use DuckDNS for the URL and Let's Encrypt for the TLS certificate (managed via acme.sh with automatic renewal). Artifacts are stored in S3 (`soen-mlflow-artifacts`) with IAM role-based access. The setup provides fairly good security without the complexity of VPNs or IP management.

### View and Enhance Your Runs

- **Browse experiments**: Compare metrics, view artifacts, search by tags
- **Add descriptions**: Click edit (✏️) to add Markdown descriptions  
- **Download artifacts**: Click any file in the Artifacts tab

## Server Management

### Check Server Status
```bash
ssh -i mlflow-key.pem ec2-user@mlflow-greatsky.duckdns.org "
  ps aux | grep mlflow | grep -v grep
  curl -s http://localhost:5000 | head -c 100
"
```

### Check SSL Certificate Status
```bash
# Check certificate expiration date
echo | openssl s_client -connect mlflow-greatsky.duckdns.org:443 -servername mlflow-greatsky.duckdns.org 2>/dev/null | openssl x509 -noout -dates

# Check auto-renewal cron job
ssh -i mlflow-key.pem ec2-user@mlflow-greatsky.duckdns.org "crontab -l | grep acme"
```

The certificate auto-renews 60 days before expiration. If you see expiration warnings, see the troubleshooting section above.

### Restart MLflow Service
```bash
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "
  pkill -f mlflow
  nohup ~/.local/bin/mlflow server \
    --backend-store-uri sqlite:///~/mlflow/mlflow.db \
    --default-artifact-root s3://soen-mlflow-artifacts/mlflow \
    --host 0.0.0.0 --port 5000 > ~/mlflow.log 2>&1 &
"
```

### View Server Logs
```bash
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "tail -50 ~/mlflow.log"
```

## Architecture

```
Training on Laptop, Remote Server or Cluster
    ↓ (metrics, params, tags)
MLflow Server (EC2)
    ↓ (artifacts: .ckpt, .soen)
S3 Bucket (soen-mlflow-artifacts)
```

## Why Use a Server vs Direct S3?

You could theoretically run MLflow UI locally and connect directly to S3, but the server approach is better for teams:

| Aspect | **Server (Current)** | Direct S3 |
|--------|---------------------|-----------|
| **Team access** | ✅ Just need server URL | ❌ Everyone needs AWS creds |
| **Security** | ✅ Centralized, IP-restricted | ❌ AWS keys on every laptop |
| **New team members** | ✅ Zero setup (just YAML config) | ❌ AWS credential setup |
| **Shared workspace** | ✅ Everyone sees same experiments | ❌ Each person runs own UI |
| **Access control** | ✅ IP restrictions, no credential sharing | ❌ Full S3 access required |
| **Maintenance** | ⚠️ Minimal server upkeep | ✅ None |

**Bottom line**: The server adds minimal overhead but provides much better team collaboration and security.

### Components Created

- **S3 Bucket**: `soen-mlflow-artifacts` (artifacts storage)
- **IAM Role**: `mlflow-ec2-role` (EC2 → S3 permissions)
- **IAM Policy**: `mlflow-artifacts-s3-access` (S3 read/write)
- **Security Group**: `mlflow-sg` (SSH + MLflow UI access)
- **EC2 Instance**: `t3.small` with 20GB disk
- **Key Pair**: `mlflow-key.pem` (SSH access)

### What Gets Logged Automatically

- **Metrics**: Loss curves, accuracy, custom metrics from training
- **Parameters**: All hyperparameters from your training config  
- **Tags**: Project info, team member, repeat/seed numbers
- **Artifacts**: Checkpoints (`.ckpt`), SOEN sidecars (`.soen`)
- **Rich descriptions**: Add Markdown text, upload images via UI

## Troubleshooting

### SSL Certificate Expired

**Symptom:** Browser shows "Your connection is not private" or "NET::ERR_CERT_DATE_INVALID" with message about expired certificate.

**Cause:** The Let's Encrypt certificate expires every 90 days. While auto-renewal is configured via acme.sh, the deployment to Nginx may fail due to permissions.

**Fix:**
```bash
ssh -i mlflow-key.pem ec2-user@mlflow-greatsky.duckdns.org "
# Check current certificate expiration
sudo openssl x509 -in /etc/nginx/ssl/nginx.crt -noout -dates

# If expired, deploy the renewed certificate from acme.sh
sudo cp ~/.acme.sh/mlflow-greatsky.duckdns.org_ecc/mlflow-greatsky.duckdns.org.key /etc/nginx/ssl/nginx.key
sudo cp ~/.acme.sh/mlflow-greatsky.duckdns.org_ecc/fullchain.cer /etc/nginx/ssl/nginx.crt
sudo systemctl reload nginx

# Verify it's fixed
sudo openssl x509 -in /etc/nginx/ssl/nginx.crt -noout -dates
"
```

The certificate renewal runs daily at 20:44 UTC (check with `crontab -l`).

### Connection Refused or Timeout

**Most common cause: Your IP address changed**

Your public IP can change daily due to dynamic ISP assignment. Check your current IP:
```bash
curl -s https://checkip.amazonaws.com
```

If it's different from yesterday, update the security group:
```bash
./update-my-ip.sh
```

**Other causes:**
- Check security group allows your IP on port 5000
- Verify MLflow process is running: `ps aux | grep mlflow`

### Disk Full
- The setup uses t3.small with 20GB disk
- Monitor with: `df -h`
- Clean old artifacts from S3 if needed

### Permission Denied (S3)
- Verify IAM role attached to EC2 instance
- Check S3 bucket policy allows the role

### Performance Issues
- Upgrade instance type in setup script (t3.medium, etc.)
- Consider RDS instead of SQLite for backend store

## Advanced Configuration

### Using RDS Backend (Recommended for Teams)

1. Create RDS Postgres instance
2. Update MLflow server command:
```bash
mlflow server \
  --backend-store-uri postgresql+psycopg2://USER:PASS@HOST:5432/mlflow \
  --default-artifact-root s3://soen-mlflow-artifacts/mlflow \
  --host 0.0.0.0 --port 5000
```

### Professional Authentication Setup

**Problem**: IP-based access doesn't scale for large teams (IPs change daily, security group gets cluttered).

**Solution**: Add authentication proxy that allows global access but requires login.

```bash
./add-auth-proxy.sh
```

This creates:
- HTTPS access from anywhere (no IP restrictions needed)
- Basic authentication (username/password)
- Let's Encrypt certificate via acme.sh (automatically renewing)

**After running, access via:**
- URL: `https://54.227.76.97` (note: HTTPS, no port)
- Username: `admin`, Password: `mlflow123`

**Update training configs:**
```yaml
logging:
  mlflow_tracking_uri: "https://admin:mlflow123@54.227.76.97"
```

### Custom Domain + HTTPS

The current setup already uses a custom domain (mlflow-greatsky.duckdns.org) with Let's Encrypt:
1. DuckDNS points to the EC2 public IP
2. acme.sh manages Let's Encrypt certificate renewal (runs daily)
3. Nginx serves HTTPS with the Let's Encrypt certificate
4. Certificate auto-renews 60 days before expiration

To use a different domain:
1. Point your DNS to the EC2 public IP
2. Run: `~/.acme.sh/acme.sh --issue -d yourdomain.com --dns dns_provider`
3. Update Nginx config `/etc/nginx/conf.d/mlflow.conf` with new domain
4. Configure acme.sh to deploy renewed certs to Nginx

### SSH Access for Admins

MLflow web access works from anywhere (authentication-based), but SSH access to the server still requires IP allowlisting:

```bash
# Check your current IP
curl -s https://checkip.amazonaws.com

# Add your IP for SSH access (port 22 only)
aws ec2 authorize-security-group-ingress \
  --group-id sg-0efd8e030d21a89d6 \
  --protocol tcp --port 22 \
  --cidr YOUR_IP/32
```

**Note**: Only needed for server administration. Regular MLflow usage requires no IP management.

## Cost Optimization

- **t3.small**: ~$15/month
- **S3 storage**: ~$0.02/GB/month
- **Data transfer**: Usually within free tier


## Security Notes

- Server only accepts connections from specified IPs
- No authentication on MLflow UI (consider Nginx auth for production)
- S3 access via IAM roles (no hardcoded keys)
- SSH key required for server access

## Integration Details

 The SOEN toolkit integration:
- Uses `SafeMLFlowLogger` to sanitize metric names (`val_loss/total` → `val_loss_total`)
- Logs artifacts via `SOENModelCheckpoint` callback
- Falls back gracefully if MLflow unavailable
- Works alongside existing TensorBoard logging

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="index.md" class="nav-home">Home</a>
</div>
