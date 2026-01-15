# MLflow Server Troubleshooting Guide

## Common Issues and Solutions

### 1. SSL Certificate Expired

**Symptoms:**
- Browser shows "Your connection is not private"
- Error: "NET::ERR_CERT_DATE_INVALID"
- Message about certificate expiring or expired

**Cause:**
Let's Encrypt certificates expire every 90 days. While acme.sh runs daily to renew, the deployment to Nginx may fail due to permissions.

**Solutions:**

Check certificate expiration:
```bash
ssh -i mlflow-key.pem ec2-user@mlflow-greatsky.duckdns.org "sudo openssl x509 -in /etc/nginx/ssl/nginx.crt -noout -dates"
```

If expired, manually deploy the renewed certificate:
```bash
ssh -i mlflow-key.pem ec2-user@mlflow-greatsky.duckdns.org "
sudo cp ~/.acme.sh/mlflow-greatsky.duckdns.org_ecc/mlflow-greatsky.duckdns.org.key /etc/nginx/ssl/nginx.key
sudo cp ~/.acme.sh/mlflow-greatsky.duckdns.org_ecc/fullchain.cer /etc/nginx/ssl/nginx.crt
sudo chmod 600 /etc/nginx/ssl/nginx.key
sudo chmod 644 /etc/nginx/ssl/nginx.crt
sudo systemctl reload nginx
"
```

Verify the fix:
```bash
echo | openssl s_client -connect mlflow-greatsky.duckdns.org:443 -servername mlflow-greatsky.duckdns.org 2>/dev/null | openssl x509 -noout -dates
```

Check auto-renewal cron:
```bash
ssh -i mlflow-key.pem ec2-user@mlflow-greatsky.duckdns.org "crontab -l | grep acme"
```

### 2. "Connection Refused" Error

**Symptoms:**
- Browser shows "ERR_CONNECTION_REFUSED" 
- Training fails with "MLflow logger setup failed"

**Solutions:**

Check if MLflow is running:
```bash
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "ps aux | grep mlflow"
```

If not running, start manually:
```bash
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "
cd ~
nohup ~/.local/bin/mlflow server \
  --backend-store-uri sqlite:///~/mlflow/mlflow.db \
  --default-artifact-root s3://soen-mlflow-artifacts/mlflow \
  --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
"
```

Check security group allows your IP:
```bash
# Get your current IP
curl -s https://checkip.amazonaws.com

# Update security group (replace sg-XXXXX and YOUR_IP)
aws ec2 authorize-security-group-ingress \
  --group-id sg-XXXXX \
  --protocol tcp --port 5000 \
  --cidr YOUR_IP/32
```

### 3. "No Space Left on Device"

**Symptoms:**
- MLflow installation fails during pip install
- Docker pulls fail with disk space errors

**Solutions:**

Check disk usage:
```bash
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "df -h"
```

If root filesystem is full:
1. Terminate current instance
2. Launch new one with larger disk (edit setup script: VolumeSize: 30)
3. Or upgrade to larger instance type (t3.medium)

Clean up space (temporary fix):
```bash
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "
sudo yum clean all
sudo docker system prune -af
rm -rf ~/.cache/pip
"
```

### 4. "Invalid Artifact Path" Error

**Symptoms:**
- Training runs but crashes at the end with MLflow artifact errors
- Error mentions checkpoint filenames

**This is already fixed** in the current integration via:
- `SafeMLFlowLogger` sanitizes metric names
- `log_model=False` disables automatic checkpoint scanning
- Manual artifact logging in `SOENModelCheckpoint`

### 5. Blank MLflow UI

**Symptoms:**
- MLflow server responds but UI shows blank page
- No experiments visible

**Solutions:**

Try direct URL with hash route:
```
http://YOUR_SERVER_IP:5000/#/experiments
```

Clear browser cache or try incognito mode.

Check if experiments exist:
```bash
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "
ls -la ~/mlflow/
sqlite3 ~/mlflow/mlflow.db 'SELECT * FROM experiments;'
"
```

### 6. S3 Permission Errors

**Symptoms:**
- MLflow starts but artifacts fail to upload
- "Access Denied" errors in logs

**Solutions:**

Verify IAM role is attached:
```bash
aws ec2 describe-instances --instance-ids YOUR_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].IamInstanceProfile'
```

Test S3 access from instance:
```bash
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "
aws s3 ls s3://soen-mlflow-artifacts/
echo 'test' | aws s3 cp - s3://soen-mlflow-artifacts/test.txt
aws s3 rm s3://soen-mlflow-artifacts/test.txt
"
```

If fails, re-attach IAM role:
```bash
aws ec2 associate-iam-instance-profile \
  --instance-id YOUR_INSTANCE_ID \
  --iam-instance-profile Name=mlflow-ec2-role
```

### 7. Training Hangs on MLflow Logging

**Symptoms:**
- Training starts but hangs when logging first metrics
- No error messages

**Solutions:**

Check MLflow server is reachable from training machine:
```bash
curl -v http://YOUR_SERVER_IP:5000/api/2.0/mlflow/experiments/list
```

Disable MLflow temporarily to isolate issue:
```yaml
logging:
  mlflow_active: false  # Temporary
```

Check for network/firewall issues between training and server.

### 8. Multiple Team Members Can't Access

**Symptoms:**
- Works for one person but not others
- "Connection timeout" for some team members

**Solutions:**

Add all team IPs to security group:
```bash
# For each team member IP
aws ec2 authorize-security-group-ingress \
  --group-id sg-XXXXX \
  --protocol tcp --port 5000 \
  --cidr TEAM_MEMBER_IP/32
```

Or use office/VPN CIDR block:
```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-XXXXX \
  --protocol tcp --port 5000 \
  --cidr 10.0.0.0/8  # Example office network
```

### 9. Server Stops After Reboot

**Symptoms:**
- MLflow works initially but stops after EC2 reboot
- Need to manually restart after instance starts

**Solutions:**

Create systemd service for auto-start:
```bash
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "
sudo tee /etc/systemd/system/mlflow.service > /dev/null <<EOF
[Unit]
Description=MLflow Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user
Environment=PATH=/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ec2-user/.local/bin/mlflow server --backend-store-uri sqlite:////home/ec2-user/mlflow/mlflow.db --default-artifact-root s3://soen-mlflow-artifacts/mlflow --host 0.0.0.0 --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
"
```

## Getting Help

### Check Server Status
```bash
# Quick health check
curl -s http://YOUR_SERVER_IP:5000/health || echo "Server not responding"

# Detailed status
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "
echo '=== System Status ==='
uptime
df -h
echo
echo '=== MLflow Process ==='
ps aux | grep mlflow | grep -v grep
echo
echo '=== Recent Logs ==='
tail -20 ~/mlflow.log
echo
echo '=== Network ==='
sudo netstat -tlnp | grep 5000
"
```

### Collect Debug Info
```bash
# Run this and share output when asking for help
ssh -i mlflow-key.pem ec2-user@YOUR_SERVER_IP "
echo '=== Instance Info ==='
curl -s http://169.254.169.254/latest/meta-data/instance-id
curl -s http://169.254.169.254/latest/meta-data/instance-type
echo
echo '=== MLflow Version ==='
~/.local/bin/mlflow --version
echo
echo '=== Python Packages ==='
pip3 list | grep -E '(mlflow|boto3|flask)'
echo
echo '=== Disk Usage ==='
df -h
echo
echo '=== Memory Usage ==='
free -h
echo
echo '=== S3 Test ==='
aws s3 ls s3://soen-mlflow-artifacts/ | head -5
"
```

### Emergency Recovery

If server is completely broken, quickest fix:
1. Terminate instance: `aws ec2 terminate-instances --instance-ids YOUR_INSTANCE_ID`
2. **DO NOT** run the setup script again (it will try to recreate existing resources)
3. Instead, manually launch a new instance with the existing IAM role and security group
4. SSH in and install MLflow manually (see setup steps in README)
5. Update team configs with new IP

The S3 data and experiments will be preserved as long as you don't delete the bucket.
