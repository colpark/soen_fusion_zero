#!/bin/bash
set -euo pipefail

# MLflow Server Setup Script
# This script creates all AWS resources needed for a shared MLflow tracking server

echo "🚀 Setting up MLflow server infrastructure..."

# Configuration
BUCKET_NAME="soen-mlflow-artifacts"
POLICY_NAME="mlflow-artifacts-s3-access"
ROLE_NAME="mlflow-ec2-role"
SG_NAME="mlflow-sg"
INSTANCE_NAME="mlflow-server"
KEY_NAME="mlflow-key"
INSTANCE_TYPE="t3.micro"
YOUR_IP=$(curl -s https://checkip.amazonaws.com)

echo "📍 Your current IP: $YOUR_IP"

# Get default VPC ID
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
echo "🏠 Using default VPC: $VPC_ID"

# 1. Create S3 bucket if it doesn't exist
echo "📦 Creating S3 bucket..."
if ! aws s3 ls "s3://$BUCKET_NAME" >/dev/null 2>&1; then
    aws s3 mb "s3://$BUCKET_NAME"
    echo "✅ Created bucket: $BUCKET_NAME"
else
    echo "✅ Bucket already exists: $BUCKET_NAME"
fi

# Create mlflow/ prefix directory
aws s3api put-object --bucket "$BUCKET_NAME" --key "mlflow/" --content-length 0 || true

# 2. Create IAM policy
echo "🔐 Creating IAM policy..."
cat > /tmp/mlflow-s3-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::$BUCKET_NAME"],
      "Condition": {
        "StringLike": {
          "s3:prefix": ["mlflow/*"]
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": ["arn:aws:s3:::$BUCKET_NAME/mlflow/*"]
    }
  ]
}
EOF

# Delete existing policy if it exists
aws iam delete-policy --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$POLICY_NAME" 2>/dev/null || true

POLICY_ARN=$(aws iam create-policy \
  --policy-name "$POLICY_NAME" \
  --policy-document file:///tmp/mlflow-s3-policy.json \
  --query 'Policy.Arn' --output text)
echo "✅ Created policy: $POLICY_ARN"

# 3. Create IAM role
echo "🔐 Creating IAM role..."
cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Delete existing role if it exists
aws iam remove-role-from-instance-profile --instance-profile-name "$ROLE_NAME" --role-name "$ROLE_NAME" 2>/dev/null || true
aws iam delete-instance-profile --instance-profile-name "$ROLE_NAME" 2>/dev/null || true
aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn "$POLICY_ARN" 2>/dev/null || true
aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore" 2>/dev/null || true
aws iam delete-role --role-name "$ROLE_NAME" 2>/dev/null || true

aws iam create-role \
  --role-name "$ROLE_NAME" \
  --assume-role-policy-document file:///tmp/trust-policy.json >/dev/null

aws iam attach-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-arn "$POLICY_ARN"

aws iam attach-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"

aws iam create-instance-profile --instance-profile-name "$ROLE_NAME" >/dev/null
aws iam add-role-to-instance-profile \
  --instance-profile-name "$ROLE_NAME" \
  --role-name "$ROLE_NAME"

echo "✅ Created role and instance profile: $ROLE_NAME"

# 4. Create security group
echo "🔥 Creating security group..."
# Delete existing SG if it exists
OLD_SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [ "$OLD_SG_ID" != "None" ]; then
    aws ec2 delete-security-group --group-id "$OLD_SG_ID" 2>/dev/null || true
fi

SG_ID=$(aws ec2 create-security-group \
  --group-name "$SG_NAME" \
  --description "Allow GreatSky team access for MLflow server" \
  --vpc-id "$VPC_ID" \
  --query 'GroupId' --output text)

# Add inbound rules
aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --ip-permissions '[
    {"IpProtocol":"tcp","FromPort":22,"ToPort":22,"IpRanges":[{"CidrIp":"'"$YOUR_IP"'/32","Description":"SSH admin"}]},
    {"IpProtocol":"tcp","FromPort":5000,"ToPort":5000,"IpRanges":[{"CidrIp":"'"$YOUR_IP"'/32","Description":"MLflow UI"}]}
  ]'

echo "✅ Created security group: $SG_ID"

# 5. Create key pair
echo "🔑 Creating key pair..."
aws ec2 delete-key-pair --key-name "$KEY_NAME" 2>/dev/null || true
aws ec2 create-key-pair --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > "$KEY_NAME.pem"
chmod 400 "$KEY_NAME.pem"
echo "✅ Created key pair: $KEY_NAME.pem"

# 6. Create user data script
cat > /tmp/mlflow-userdata.sh <<'EOF'
#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/user-data.log) 2>&1

# Update system
yum update -y

# Install Docker
yum install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user

# Create mlflow directory
mkdir -p /home/ec2-user/mlflow
chown ec2-user:ec2-user /home/ec2-user/mlflow

# Wait for Docker to be ready
sleep 10

# Run MLflow server
docker run -d --name mlflow --restart always -p 5000:5000 \
  -v /home/ec2-user/mlflow:/mlflow \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root s3://soen-mlflow-artifacts/mlflow \
    --host 0.0.0.0 --port 5000

# Create systemd service for MLflow (backup)
cat > /etc/systemd/system/mlflow.service <<'MLFLOW_EOF'
[Unit]
Description=MLflow Server
After=docker.service
Requires=docker.service

[Service]
User=ec2-user
Restart=always
RestartSec=10
ExecStartPre=/usr/bin/docker rm -f mlflow || true
ExecStart=/usr/bin/docker run --name mlflow -p 5000:5000 \
  -v /home/ec2-user/mlflow:/mlflow \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root s3://soen-mlflow-artifacts/mlflow \
    --host 0.0.0.0 --port 5000
ExecStop=/usr/bin/docker stop mlflow

[Install]
WantedBy=multi-user.target
MLFLOW_EOF

systemctl daemon-reload
systemctl enable mlflow

echo "MLflow server setup complete" > /home/ec2-user/setup-complete.txt
chown ec2-user:ec2-user /home/ec2-user/setup-complete.txt
EOF

# 7. Get latest Amazon Linux 2023 AMI
echo "🖼️ Finding latest Amazon Linux 2023 AMI..."
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-*-kernel-6.1-x86_64" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)
echo "✅ Using AMI: $AMI_ID"

# 8. Launch instance
echo "🚀 Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --iam-instance-profile "Name=$ROLE_NAME" \
  --security-group-ids "$SG_ID" \
  --user-data file:///tmp/mlflow-userdata.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "✅ Launched instance: $INSTANCE_ID"

# Wait for instance to be running
echo "⏳ Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "✅ Instance is running!"
echo ""
echo "🎉 MLflow server setup complete!"
echo ""
echo "📋 Summary:"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  SSH Key: $KEY_NAME.pem"
echo "  MLflow UI: http://$PUBLIC_IP:5000"
echo ""
echo "🔧 Next steps:"
echo "  1. Wait 2-3 minutes for MLflow to start"
echo "  2. Open http://$PUBLIC_IP:5000 in your browser"
echo "  3. Set your training YAML:"
echo "     mlflow_tracking_uri: \"http://$PUBLIC_IP:5000\""
echo ""
echo "🐛 To debug:"
echo "  ssh -i $KEY_NAME.pem ec2-user@$PUBLIC_IP"
echo "  docker logs mlflow"
echo ""

# Clean up temp files
rm -f /tmp/mlflow-s3-policy.json /tmp/trust-policy.json /tmp/mlflow-userdata.sh

echo "✨ All done!"
