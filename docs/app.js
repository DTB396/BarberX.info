const $q = document.getElementById("q");
const $results = document.getElementById("results");
const $meta = document.getElementById("meta");

let data = [];

// Configuration constants
const MAX_RESULTS = 50;  // Maximum number of search results to display
const MAX_TAGS_DISPLAYED = 6;  // Maximum number of tags to show per item
const PHRASE_MATCH_SCORE = 10;  // Score bonus for exact phrase match
const MAX_TOKEN_HITS = 20;  // Maximum score contribution per token
const DEBOUNCE_DELAY_MS = 150;  // Delay in ms before processing search input

/**
 * Scores how well a query matches against text.
 * 
 * Scoring algorithm:
 * 1. Exact phrase match: +10 points (PHRASE_MATCH_SCORE)
 * 2. Token-based matching: For each token in the query, count occurrences
 *    in the text and add to score (capped at 20 points per token)
 * 
 * @param {string} query - The search query
 * @param {string} text - The text to search within
 * @returns {number} - The match score (higher is better, 0 = no match)
 */
function scoreMatch(query, text) {
  if (!query) return 0;
  const q = query.toLowerCase().trim();
  if (!q) return 0;

  const t = (text || "").toLowerCase();
  if (!t) return 0;

  // Simple scoring: phrase > token hits
  let score = 0;
  // Award points for exact phrase match
  if (t.includes(q)) score += PHRASE_MATCH_SCORE;

  // Award points for individual token matches
  const tokens = q.split(/\s+/).filter(Boolean);
  for (const tok of tokens) {
    const hits = t.split(tok).length - 1;
    score += Math.min(hits, MAX_TOKEN_HITS);
  }
  return score;
}

function render(query) {
  const q = (query || "").trim();
  const rows = data
    .map(item => {
      const text = item.text || item.snippet || "";
      const s = scoreMatch(q, `${item.title}\n${item.caseId}\n${(item.tags || []).join(" ")}\n${text}`);
      return { item, s };
    })
    .filter(x => q ? x.s > 0 : true)
    .sort((a,b) => b.s - a.s)
    .slice(0, MAX_RESULTS);

  $meta.textContent = data.length
    ? `${rows.length} result(s) shown • ${data.length} document(s) indexed`
    : "Loading index…";

  // Update ARIA live region for accessibility
  if (!$results.hasAttribute('aria-live')) {
    $results.setAttribute('aria-live', 'polite');
    $results.setAttribute('aria-relevant', 'additions removals');
  }

  $results.innerHTML = rows.map(({item, s}) => {
    const allTags = item.tags || [];
    const tags = allTags.slice(0, MAX_TAGS_DISPLAYED);
    const hasMoreTags = allTags.length > MAX_TAGS_DISPLAYED;
    const ocr = item.ocrNeeded ? `<span class="badge">OCR needed</span>` : "";
    const ok = item.ok ? "" : `<span class="badge">Index error</span>`;
    const err = item.error ? `<div class="snip">Index error: ${item.error}</div>` : "";
    return `
      <div class="card">
        <h3>${escapeHtml(item.title || item.caseId || "Untitled")}</h3>
        <div class="badges">
          <span class="badge">${escapeHtml(item.caseId || "")}</span>
          ${tags.map(t => `<span class="badge">${escapeHtml(t)}</span>`).join("")}
          ${hasMoreTags ? `<span class="badge">+${allTags.length - MAX_TAGS_DISPLAYED} more</span>` : ""}
          ${ocr}
          ${ok}
          ${q ? `<span class="badge">score: ${s}</span>` : ""}
        </div>
        ${item.snippet ? `<div class="snip">${escapeHtml(item.snippet)}</div>` : ""}
        ${err}
        <div class="actions">
          <a class="btn primary" href="${sanitizeUrl(item.url)}" target="_blank" rel="noopener">Open PDF</a>
        </div>
      </div>
    `;
  }).join("");
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"'`]/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#x27;","`":"&#x60;"
  }[c]));
}

/**
 * Sanitizes a URL to ensure it uses a safe protocol.
 * Only allows http:, https:, and relative URLs.
 * 
 * @param {string} url - The URL to sanitize
 * @returns {string} - The sanitized URL or '#' if invalid
 */
function sanitizeUrl(url) {
  if (!url) return "#";
  const urlStr = String(url).trim();
  if (!urlStr) return "#";
  
  // Allow relative URLs (starting with ./ or /)
  if (urlStr.startsWith("./") || urlStr.startsWith("/")) {
    return urlStr;
  }
  
  // Check for safe protocols
  try {
    const parsed = new URL(urlStr, window.location.href);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return urlStr;
    }
  } catch (e) {
    // Invalid URL, return safe default
  }
  
  return "#";
}

/**
 * Creates a debounced version of a function that delays execution
 * until after the specified delay has elapsed since the last call.
 * 
 * @param {Function} func - The function to debounce
 * @param {number} delay - The delay in milliseconds
 * @returns {Function} - The debounced function
 */
function debounce(func, delay) {
  let timeoutId;
  return function(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(this, args), delay);
  };
}

/**
 * Formats an error object into a readable message.
 * 
 * @param {Error|any} err - The error object
 * @param {string} prefix - The prefix for the error message
 * @returns {string} - The formatted error message
 */
function formatError(err, prefix) {
  return prefix + ": " + (err && err.message ? err.message : String(err));
}

async function init() {
  // Set up accessibility attributes
  if ($q && !$q.hasAttribute('aria-label')) {
    $q.setAttribute('aria-label', 'Search documents');
    $q.setAttribute('role', 'searchbox');
    $q.setAttribute('aria-controls', 'results');
  }
  
  let res;
  try {
    res = await fetch("./data/index.json");
  } catch (err) {
    throw new Error(formatError(err, "Failed to fetch index.json"));
  }

  if (!res.ok) {
    throw new Error(`Failed to fetch index.json: ${res.status} ${res.statusText}`);
  }

  try {
    data = await res.json();
  } catch (err) {
    throw new Error(formatError(err, "Failed to parse index.json"));
  }
  
  $meta.textContent = `${data.length} document(s) indexed`;
  render("");
}

$q.addEventListener("input", debounce(() => render($q.value), DEBOUNCE_DELAY_MS));
init().catch(err => {
  $meta.textContent = "Failed to load index.json";
  $results.innerHTML = `<div class="card"><div class="snip">${escapeHtml(String(err))}</div></div>`;
});
