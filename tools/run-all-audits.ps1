#!/usr/bin/env pwsh
# Automated repo audit and analysis script for Tillerstead.com
# Runs installed tools and configures them for this repo

# Start local server (Five Server)
Write-Host "Starting Five Server for live preview..."
code --install-extension yandeu.five-server
code --force --extension yandeu.five-server

# Run SEO Audit for HTML
Write-Host "Running SEO Audit for HTML..."
code --install-extension alexiscarbillet.seo-audit-for-html
code --force --extension alexiscarbillet.seo-audit-for-html

# Run RondsDoctor audit
Write-Host "Running RondsDoctor audit..."
code --install-extension renjieyin.ronds-doctor
code --force --extension renjieyin.ronds-doctor

# Run Iceworks Doctor audit
Write-Host "Running Iceworks Doctor audit..."
code --install-extension iceworks-team.iceworks-doctor
code --force --extension iceworks-team.iceworks-doctor

# Open repo in VS Code for interactive extension use
Write-Host "Opening repo in VS Code for interactive analysis..."
code .

Write-Host "All tools installed and ready. Use extension sidebars and commands for full audits."
