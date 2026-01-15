#!/bin/bash
# Add Nginx basic auth proxy (simpler than Caddy for Amazon Linux)

set -euo pipefail

echo "🔐 Adding Nginx authentication proxy..."

ssh -o StrictHostKeyChecking=no -i mlflow-key.pem ec2-user@54.227.76.97 "
# Install Nginx and httpd-tools (for htpasswd)
sudo yum install -y nginx httpd-tools

# Create basic auth password file
sudo htpasswd -cb /etc/nginx/.htpasswd admin mlflow123

# Create Nginx config
sudo tee /etc/nginx/conf.d/mlflow.conf > /dev/null <<'EOF'
server {
    listen 80;
    server_name _;
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    server_name _;
    
    # Let's Encrypt certificate (managed by acme.sh)
    ssl_certificate /etc/nginx/ssl/nginx.crt;
    ssl_certificate_key /etc/nginx/ssl/nginx.key;
    
    # Basic authentication
    auth_basic \"MLflow Access\";
    auth_basic_user_file /etc/nginx/.htpasswd;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Create self-signed certificate (temporary - replace with Let's Encrypt via acme.sh)
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/nginx.key \
  -out /etc/nginx/ssl/nginx.crt \
  -subj '/C=US/ST=State/L=City/O=Organization/CN=mlflow-server'
# NOTE: For production, install acme.sh and get Let's Encrypt certificate:
# curl https://get.acme.sh | sh
# ~/.acme.sh/acme.sh --issue -d your-domain.duckdns.org --dns dns_duckdns
# Then configure auto-deployment to /etc/nginx/ssl/

# Enable and start Nginx
sudo systemctl enable --now nginx
sudo systemctl status nginx --no-pager

echo '✅ Nginx proxy setup complete!'
echo 'Access: https://54.227.76.97'
echo 'Username: admin'
echo 'Password: mlflow123'
"

echo ""
echo "🎉 Professional authentication setup complete!"
echo ""
echo "📋 Team access:"
echo "  URL: https://54.227.76.97"
echo "  Username: admin"
echo "  Password: mlflow123"
echo ""
echo "🔧 Update training configs:"
echo "  mlflow_tracking_uri: \"https://admin:mlflow123@54.227.76.97\""
echo ""
echo "✅ Benefits:"
echo "  - No more daily IP updates needed"
echo "  - Secure authentication for team"
echo "  - HTTPS encryption"
echo "  - Access from anywhere"
