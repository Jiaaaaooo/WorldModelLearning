# World Model Learning Roadmap 🌍

> **Current Status**: Environment `lj_wml` created (dependencies pending).  
> **Hardware**: 2x RTX 3090 (Local Debug) | 4x A100 (Cluster Training)

---

## 📝 Git Configuration Log (Server Setup Tutorial)
*Recorded on 2025-12-08. Use this guide when setting up a new server.*

### 1. Identity Configuration
Tell Git who you are (Global Config).
```bash
git config --global user.name "Jiaaaaooo"
git config --global user.email "jia_aa@126.com"
# Verify config
git config --global --list
```

### 2. SSH Authentication (No-Browser Login)
Since the 3090 server has no GUI, use SSH keys to push code to GitHub.

**Step 1: Generate Key**
```bash
ssh-keygen -t ed25519 -C "jia_aa@126.com"
# Press Enter for all prompts (default path, no passphrase)
```

**Step 2: Get Public Key**
```bash
cat ~/.ssh/id_ed25519.pub
```
*Action*: Copy the output (starts with `ssh-ed25519`) to **GitHub -> Settings -> SSH and GPG keys -> New SSH key**.

### 3. Repository Initialization
How to start a new project from scratch.
```bash
# Initialize
git init
git branch -M main

# Ignore large files (Important for AI models!)
vim .gitignore  # Add *.pth, data/, etc.

# Link to Remote (Run once)
git remote add origin git@github.com:Jiaaaaooo/WorldModelLearning.git
```

### 4. Daily Workflow (Cheatsheet)
```bash
# 1. Save changes locally
git add .
git commit -m "update: detailed description"

# 2. Upload to GitHub
git push
```

---
*Maintained by @Jiaaaaooo*
