#!/bin/bash

echo "🚀 Preparing for deployment..."

# Copy simplified requirements for deployment
echo "📦 Creating deployment requirements..."
cp requirements-deploy.txt requirements.txt

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << EOF
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
*.sqlite3
.env
.vscode/
hf_cache/
indices/
ssl/
mx
fitz
server_log.txt
EOF
fi

echo "✅ Deployment files ready!"
echo ""
echo "📋 Next steps:"
echo "1. Create a GitHub repository"
echo "2. Push your code:"
echo "   git add ."
echo "   git commit -m 'Initial deployment'"
echo "   git push origin main"
echo ""
echo "3. Go to https://railway.app and deploy from GitHub"
echo "4. Your app will be live at: https://your-app-name.railway.app"
echo ""
echo "🎯 See DEPLOYMENT.md for detailed instructions!" 