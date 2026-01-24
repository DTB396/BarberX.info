#!/bin/bash
# Netlify build script - runs before Jekyll build

echo "🔧 BarberX.info Build Started"
echo "📦 Ruby version: $(ruby -v)"
echo "💎 Bundler version: $(bundle -v)"
echo "🌿 Jekyll version: $(bundle exec jekyll -v)"

# Ensure all dependencies are installed
echo "📥 Installing dependencies..."
bundle install --quiet

echo "✅ Build environment ready"
