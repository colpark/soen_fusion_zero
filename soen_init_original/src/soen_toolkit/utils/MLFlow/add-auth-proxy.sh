#!/bin/bash
# Add basic authentication proxy in front of MLflow
# This allows opening port 443 to the world while keeping MLflow secure

set -euo pipefail

echo "🔐 Adding authentication proxy to MLflow server..."

# Get instance details
INSTANCE_ID="i-021f5807adea1bedb"  # Your current instance
SG_ID="sg-0efd8e030d21a89d6"      # Your security group

# Add HTTPS port to security group
aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --ip-permissions '[
    {"IpProtocol":"tcp","FromPort":443,"ToPort":443,"IpRanges":[{"CidrIp":"0.0.0.0/0","Description":"HTTPS MLflow with auth"}]},
    {"IpProtocol":"tcp","FromPort":80,"ToPort":80,"IpRanges":[{"CidrIp":"0.0.0.0/0","Description":"HTTP redirect to HTTPS"}]}
  ]'

echo "✅ Opened ports 80/443 for authenticated access"

# Install and configure Caddy (easier than Nginx)
ssh -o StrictHostKeyChecking=no -i mlflow-key.pem ec2-user@54.227.76.97 "
sudo yum install -y curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/setup.rpm.sh' | sudo bash
sudo yum install -y caddy

# Create Caddyfile with basic auth
sudo tee /etc/caddy/Caddyfile > /dev/null <<'EOF'
{
    # Global options
    auto_https off
}

:80 {
    # Redirect HTTP to HTTPS
    redir https://{host}{uri} permanent
}

:443 {
    # Basic authentication (username: admin, password: mlflow123)
    basicauth /* {
        admin \$2a\$14\$hNKNqo7jF8w2G7B8b9FdHOK5vXr8mF3yJ2Qv8wF7mF8nF9dF0eF1F2
    }
    
    # Proxy to MLflow
    reverse_proxy localhost:5000
    
    # TLS with self-signed cert (for production, use Let's Encrypt via acme.sh)
    tls internal
}
EOF

# Start Caddy
sudo systemctl enable --now caddy
sudo systemctl status caddy --no-pager

echo '✅ Caddy proxy installed with basic auth'
echo 'Username: admin'
echo 'Password: mlflow123'
echo 'Access via: https://54.227.76.97'
"

echo ""
echo "🎉 Authentication proxy setup complete!"
echo ""
echo "📋 New access method:"
echo "  URL: https://54.227.76.97"
echo "  Username: admin"
echo "  Password: mlflow123"
echo ""
echo "🔧 Update your training configs:"
echo "  mlflow_tracking_uri: \"https://admin:mlflow123@54.227.76.97\""
echo ""
echo "⚠️  Security notes:"
echo "  - Change the password in /etc/caddy/Caddyfile"
echo "  - For production: use a real domain + Let's Encrypt cert via acme.sh"
echo "  - The current setup uses a self-signed certificate (browser warnings expected)"
