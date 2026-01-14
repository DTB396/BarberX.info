#!/usr/bin/env bash
set -euo pipefail

# Re-run failed workflow runs with AI-powered error analysis and patching.
# Usage:
#   bash scripts/rerun_failed_workflows.sh [--analyze] [--auto-patch]
# Options:
#   --analyze      Fetch and analyze error logs with AI assistance
#   --auto-patch   Automatically apply safe fixes (requires --analyze)
#   --no-rerun     Skip rerunning workflows (analysis only)
# Environment:
#   REPO="owner/repo"           Override detected repository
#   OPENAI_API_KEY="sk-..."     Enable AI-powered error analysis

REPO="${REPO:-}"
LIMIT="${LIMIT:-50}"   # number of recent runs to scan
ANALYZE_ERRORS=false
AUTO_PATCH=false
SKIP_RERUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --analyze) ANALYZE_ERRORS=true; shift ;;
    --auto-patch) AUTO_PATCH=true; ANALYZE_ERRORS=true; shift ;;
    --no-rerun) SKIP_RERUN=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI 'gh' not found. Install it, then run: gh auth login"
  exit 1
fi

if [[ -z "$REPO" ]]; then
  # infer owner/repo from git remote
  REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"
  if [[ -z "$REMOTE_URL" ]]; then
    echo "ERROR: Could not infer repo. Set REPO=owner/repo"
    exit 1
  fi
  # supports git@github.com:owner/repo.git or https://github.com/owner/repo.git
  REPO="$(echo "$REMOTE_URL" | sed -E 's#(git@github.com:|https://github.com/)([^/]+/[^/.]+)(\.git)?#\2#')"
fi

echo "Repo: $REPO"
echo "Scanning last $LIMIT runs for failures..."

# Get failed runs with metadata
FAILED_RUNS="$(gh run list -R "$REPO" --limit "$LIMIT" --json databaseId,conclusion,status,name,workflowName,headBranch,event \
  --jq '.[] | select((.conclusion=="failure") or (.conclusion=="cancelled") or (.conclusion=="timed_out")) | @json')"

if [[ -z "$FAILED_RUNS" ]]; then
  echo "No failed/cancelled/timed_out runs found in last $LIMIT."
  exit 0
fi

RUN_COUNT=$(echo "$FAILED_RUNS" | wc -l | tr -d ' ')
echo "Found $RUN_COUNT failed run(s)"

# Error analysis
if [[ "$ANALYZE_ERRORS" == true ]]; then
  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "WORKFLOW ERROR ANALYSIS"
  echo "════════════════════════════════════════════════════════════════"
  
  REPORT_DIR="reports/workflow-errors"
  mkdir -p "$REPORT_DIR"
  REPORT_FILE="$REPORT_DIR/analysis-$(date +%Y%m%d-%H%M%S).md"
  
  echo "# Workflow Error Analysis Report" > "$REPORT_FILE"
  echo "Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "$REPORT_FILE"
  echo "Repository: $REPO" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"
  
  PATCH_COUNT=0
  ERROR_PATTERNS=()
  
  while IFS= read -r run_json; do
    [[ -z "$run_json" ]] && continue
    
    RUN_ID=$(echo "$run_json" | jq -r '.databaseId')
    WORKFLOW=$(echo "$run_json" | jq -r '.workflowName')
    BRANCH=$(echo "$run_json" | jq -r '.headBranch')
    
    echo
    echo "─────────────────────────────────────────────────────────────────"
    echo "Run #$RUN_ID: $WORKFLOW (branch: $BRANCH)"
    echo "─────────────────────────────────────────────────────────────────"
    
    # Fetch logs
    echo "Fetching logs..."
    LOG_FILE="$REPORT_DIR/run-$RUN_ID.log"
    if gh run view "$RUN_ID" -R "$REPO" --log > "$LOG_FILE" 2>&1; then
      LOG_SIZE=$(wc -l < "$LOG_FILE" | tr -d ' ')
      echo "  ✓ Downloaded $LOG_SIZE lines"
      
      # Extract errors
      ERRORS=$(grep -E "(ERROR|FAILED|Error:|error:|✗|❌)" "$LOG_FILE" | head -20 || true)
      
      if [[ -n "$ERRORS" ]]; then
        echo "  Found errors:"
        echo "$ERRORS" | sed 's/^/    /'
        
        # Append to report
        echo "## Run #$RUN_ID: $WORKFLOW" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "**Branch:** $BRANCH" >> "$REPORT_FILE"
        echo "**Status:** $(echo "$run_json" | jq -r '.conclusion')" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "### Error Summary" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "$ERRORS" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        
        # Analyze error patterns
        if echo "$ERRORS" | grep -qi "merge conflict"; then
          echo "  ⚠️  MERGE CONFLICT detected"
          ERROR_PATTERNS+=("merge_conflict")
          echo "### Recommended Fix: Merge Conflict" >> "$REPORT_FILE"
          echo "Run: \`git status\` and resolve conflicts manually." >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          
        elif echo "$ERRORS" | grep -qi "404.*not found"; then
          echo "  ⚠️  BROKEN LINK/RESOURCE detected"
          ERROR_PATTERNS+=("broken_link")
          echo "### Recommended Fix: Broken Links" >> "$REPORT_FILE"
          echo "Run: \`node scripts/check-site-links.js\` and fix broken references." >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          
        elif echo "$ERRORS" | grep -qi "pdf.*not found\|missing.*pdf"; then
          echo "  ⚠️  MISSING PDF detected"
          ERROR_PATTERNS+=("missing_pdf")
          echo "### Recommended Fix: Missing PDFs" >> "$REPORT_FILE"
          echo "Run: \`node scripts/check-pdf-links.js\` and verify all docket entries." >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          
        elif echo "$ERRORS" | grep -qi "yaml.*invalid\|yaml.*error"; then
          echo "  ⚠️  YAML SYNTAX ERROR detected"
          ERROR_PATTERNS+=("yaml_error")
          echo "### Recommended Fix: YAML Syntax" >> "$REPORT_FILE"
          echo "Validate YAML files: \`find _data -name '*.yml' -exec js-yaml {} \\;\`" >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          
        elif echo "$ERRORS" | grep -qi "liquid.*error\|liquid syntax"; then
          echo "  ⚠️  LIQUID TEMPLATE ERROR detected"
          ERROR_PATTERNS+=("liquid_error")
          echo "### Recommended Fix: Liquid Template" >> "$REPORT_FILE"
          echo "Check template syntax in \`_layouts/\` and \`_includes/\`." >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          
        elif echo "$ERRORS" | grep -qi "build.*failed\|jekyll.*error"; then
          echo "  ⚠️  JEKYLL BUILD ERROR detected"
          ERROR_PATTERNS+=("jekyll_error")
          echo "### Recommended Fix: Jekyll Build" >> "$REPORT_FILE"
          echo "Test locally: \`bundle exec jekyll build --trace\`" >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
          
        else
          echo "  ℹ️  Generic error - see logs for details"
          ERROR_PATTERNS+=("generic")
          echo "### Recommended Fix: Manual Review" >> "$REPORT_FILE"
          echo "Review full logs: \`gh run view $RUN_ID -R $REPO --log\`" >> "$REPORT_FILE"
          echo "" >> "$REPORT_FILE"
        fi
        
        # AI-powered analysis (if API key available)
        if [[ -n "${OPENAI_API_KEY:-}" ]]; then
          echo "  🤖 Requesting AI analysis..."
          
          AI_ANALYSIS=$(node -e "
            const OpenAI = require('openai');
            const fs = require('fs');
            const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
            
            (async () => {
              const errors = \`$ERRORS\`;
              const workflow = '$WORKFLOW';
              
              const response = await openai.chat.completions.create({
                model: 'gpt-4',
                messages: [
                  {
                    role: 'system',
                    content: 'You are a GitHub Actions workflow expert. Analyze errors and provide concise, actionable fixes following high-end web development and CI/CD best practices. Keep responses under 200 words.'
                  },
                  {
                    role: 'user',
                    content: \`Workflow: \${workflow}\n\nErrors:\n\${errors}\n\nProvide: 1) Root cause, 2) Specific fix, 3) Prevention strategy.\`
                  }
                ],
                temperature: 0.3,
                max_tokens: 400
              });
              
              console.log(response.choices[0].message.content);
            })();
          " 2>/dev/null || echo "AI analysis failed")
          
          if [[ "$AI_ANALYSIS" != "AI analysis failed" ]]; then
            echo "  ✓ AI analysis complete"
            echo "### AI-Powered Analysis" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            echo "$AI_ANALYSIS" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
          fi
        fi
        
        # Auto-patch (if enabled and safe pattern detected)
        if [[ "$AUTO_PATCH" == true ]]; then
          PATCHED=false
          
          case "${ERROR_PATTERNS[-1]}" in
            broken_link)
              echo "  🔧 Auto-patching broken links..."
              if node scripts/check-site-links.js --fix 2>/dev/null; then
                echo "  ✓ Links patched"
                PATCHED=true
                ((PATCH_COUNT++))
              fi
              ;;
            missing_pdf)
              echo "  🔧 Auto-patching PDF references..."
              if node scripts/repair-docket-file-links-to-filings.js 2>/dev/null; then
                echo "  ✓ PDF paths fixed"
                PATCHED=true
                ((PATCH_COUNT++))
              fi
              ;;
            yaml_error)
              echo "  ⚠️  YAML errors require manual review (skipping auto-patch)"
              ;;
            *)
              echo "  ℹ️  No auto-patch available for this error type"
              ;;
          esac
          
          if [[ "$PATCHED" == true ]]; then
            echo "### Auto-Patch Applied" >> "$REPORT_FILE"
            echo "✓ Automated fix applied successfully" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
          fi
        fi
        
      else
        echo "  ℹ️  No errors found in logs (may be infrastructure issue)"
      fi
    else
      echo "  ✗ Failed to fetch logs"
    fi
    
    echo "" >> "$REPORT_FILE"
    
  done <<< "$FAILED_RUNS"
  
  # Summary
  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "ANALYSIS SUMMARY"
  echo "════════════════════════════════════════════════════════════════"
  echo "Total failed runs: $RUN_COUNT"
  
  # Count unique patterns
  declare -A PATTERN_COUNTS
  for pattern in "${ERROR_PATTERNS[@]}"; do
    ((PATTERN_COUNTS[$pattern]++)) || true
  done
  
  if [ ${#PATTERN_COUNTS[@]} -gt 0 ]; then
    echo "Error patterns detected:"
    for pattern in "${!PATTERN_COUNTS[@]}"; do
      echo "  - $pattern: ${PATTERN_COUNTS[$pattern]}"
    done
  else
    echo "Error patterns detected: (none classified)"
  fi
  
  if [[ "$AUTO_PATCH" == true ]]; then
    echo "Auto-patches applied: $PATCH_COUNT"
  fi
  
  echo
  echo "📄 Full report saved: $REPORT_FILE"
  
  # Commit patches if any were applied
  if [[ $PATCH_COUNT -gt 0 ]]; then
    echo
    echo "Committing automated fixes..."
    git add -A
    git commit -m "fix(workflows): auto-patch $PATCH_COUNT error$([ $PATCH_COUNT -gt 1 ] && echo 's')

Automated fixes applied by workflow error analyzer:
$(for pattern in "${!PATTERN_COUNTS[@]}"; do echo "- Fixed: $pattern"; done)

See: $REPORT_FILE" || echo "  ℹ️  No changes to commit"
  fi
fi

# Re-run workflows
if [[ "$SKIP_RERUN" == false ]]; then
  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "RE-RUNNING FAILED WORKFLOWS"
  echo "════════════════════════════════════════════════════════════════"
  
  while IFS= read -r run_json; do
    [[ -z "$run_json" ]] && continue
    RUN_ID=$(echo "$run_json" | jq -r '.databaseId')
    WORKFLOW=$(echo "$run_json" | jq -r '.workflowName')
    echo "Re-running #$RUN_ID: $WORKFLOW"
    gh run rerun "$RUN_ID" -R "$REPO" --failed || echo "  ⚠️  Rerun failed for #$RUN_ID"
  done <<< "$FAILED_RUNS"
  
  echo
  echo "✓ Rerun commands issued"
else
  echo
  echo "ℹ️  Skipping workflow reruns (--no-rerun specified)"
fi

echo
echo "Done."
