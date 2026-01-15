#!/bin/bash
# Script to automatically update MLflow security group with your current IP

set -euo pipefail

SECURITY_GROUP_ID="sg-0efd8e030d21a89d6"
CURRENT_IP=$(curl -s https://checkip.amazonaws.com)

echo "Current IP: $CURRENT_IP"

# Add current IP for both SSH and MLflow
aws ec2 authorize-security-group-ingress \
  --group-id "$SECURITY_GROUP_ID" \
  --ip-permissions '[
    {"IpProtocol":"tcp","FromPort":22,"ToPort":22,"IpRanges":[{"CidrIp":"'"$CURRENT_IP"'/32","Description":"SSH - auto-updated"}]},
    {"IpProtocol":"tcp","FromPort":5000,"ToPort":5000,"IpRanges":[{"CidrIp":"'"$CURRENT_IP"'/32","Description":"MLflow UI - auto-updated"}]}
  ]' 2>/dev/null && echo "✅ Added $CURRENT_IP to security group" || echo "ℹ️  IP $CURRENT_IP may already be allowed"

echo "MLflow should now be accessible at: http://54.227.76.97:5000"
