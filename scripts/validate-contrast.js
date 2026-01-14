#!/usr/bin/env node
/**
 * Contrast Validation Script
 * Validates that all brand color combinations meet WCAG AA standards
 */

// Brand color palette from _variables.css
const colors = {
  // Stone (Light mode)
  stone50: { r: 220, g: 217, b: 210 },
  stone100: { r: 200, g: 197, b: 190 },
  stone200: { r: 189, g: 182, b: 170 },
  
  // Navy/Ink (Dark mode)
  navy950: { r: 5, g: 13, b: 28 },
  navy900: { r: 10, g: 27, b: 50 },
  navy800: { r: 15, g: 23, b: 42 },
  ink900: { r: 28, g: 27, b: 25 },
  ink700: { r: 58, g: 56, b: 52 },
  
  // Emerald (Primary brand)
  emerald600: { r: 16, g: 92, b: 74 },
  emerald500: { r: 1, g: 138, b: 106 },
  emerald400: { r: 36, g: 181, b: 138 },
  
  // Brass/Gold (Secondary brand)
  brass600: { r: 184, g: 138, b: 57 },
  brass500: { r: 212, g: 165, b: 116 },
  brass400: { r: 160, g: 122, b: 50 },
  
  // Cream/Highlight
  cream100: { r: 254, g: 243, b: 199 },
  cream50: { r: 249, g: 250, b: 251 },
  
  // Muted
  muted400: { r: 168, g: 162, b: 158 },
  muted300: { r: 203, g: 213, b: 225 },
  
  // Common
  white: { r: 255, g: 255, b: 255 },
  black: { r: 0, g: 0, b: 0 }
};

/**
 * Convert RGB component to linear value
 */
function toLinear(component) {
  const c = component / 255;
  return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/**
 * Calculate relative luminance
 */
function getLuminance({ r, g, b }) {
  const red = toLinear(r);
  const green = toLinear(g);
  const blue = toLinear(b);
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

/**
 * Calculate contrast ratio
 */
function getContrastRatio(color1, color2) {
  const lum1 = getLuminance(color1);
  const lum2 = getLuminance(color2);
  const lighter = Math.max(lum1, lum2);
  const darker = Math.min(lum1, lum2);
  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Check if contrast meets WCAG
 */
function meetsWCAG(ratio, level = 'AA', size = 'normal') {
  const requirements = {
    AA: { normal: 4.5, large: 3.0 },
    AAA: { normal: 7.0, large: 4.5 }
  };
  return ratio >= requirements[level][size];
}

/**
 * Format contrast result
 */
function formatResult(fg, bg, ratio) {
  const aa = meetsWCAG(ratio, 'AA', 'normal');
  const aaa = meetsWCAG(ratio, 'AAA', 'normal');
  const status = aaa ? '✓✓✓' : aa ? '✓✓ ' : '✗  ';
  const ratioStr = ratio.toFixed(2).padStart(5);
  const fgStr = fg.padEnd(15);
  const bgStr = bg.padEnd(15);
  const aaStr = aa ? 'AA ' : '-- ';
  const aaaStr = aaa ? 'AAA' : '---';
  return `${status} ${ratioStr}:1 | ${fgStr} on ${bgStr} | ${aaStr} ${aaaStr}`;
}

console.log('\n='.repeat(80));
console.log('Faith Frontier Contrast Validation');
console.log('='.repeat(80));
console.log('\nWCAG Standards: AA = 4.5:1 (normal), 3.0:1 (large) | AAA = 7.0:1 (normal), 4.5:1 (large)');
console.log('✓✓✓ = AAA | ✓✓  = AA | ✗   = Fails\n');

// Light backgrounds with dark text
console.log('\n--- Light Backgrounds ---');
const lightBgs = ['white', 'cream50', 'stone50', 'stone100'];
const darkTexts = ['ink900', 'ink700', 'emerald600', 'brass400'];

lightBgs.forEach(bg => {
  darkTexts.forEach(fg => {
    const ratio = getContrastRatio(colors[fg], colors[bg]);
    console.log(formatResult(fg, bg, ratio));
  });
});

// Dark backgrounds with light text
console.log('\n--- Dark Backgrounds ---');
const darkBgs = ['navy950', 'navy900', 'navy800', 'ink900'];
const lightTexts = ['cream50', 'muted300', 'emerald400', 'brass500'];

darkBgs.forEach(bg => {
  lightTexts.forEach(fg => {
    const ratio = getContrastRatio(colors[fg], colors[bg]);
    console.log(formatResult(fg, bg, ratio));
  });
});

// Brand color backgrounds
console.log('\n--- Emerald Backgrounds ---');
['cream50', 'cream100', 'white'].forEach(fg => {
  const ratio = getContrastRatio(colors[fg], colors.emerald600);
  console.log(formatResult(fg, 'emerald600', ratio));
});

console.log('\n--- Brass Backgrounds ---');
['ink900', 'navy900', 'black'].forEach(fg => {
  const ratio = getContrastRatio(colors[fg], colors.brass500);
  console.log(formatResult(fg, 'brass500', ratio));
});

// Problematic combinations (should all fail)
console.log('\n--- Known Issues (Should Fail) ---');
const badCombos = [
  ['ink900', 'navy950'],
  ['white', 'cream50'],
  ['emerald600', 'navy950'],
  ['brass500', 'cream50']
];

badCombos.forEach(([fg, bg]) => {
  const ratio = getContrastRatio(colors[fg], colors[bg]);
  console.log(formatResult(fg, bg, ratio));
});

console.log('\n' + '='.repeat(80));
console.log('✓ Validation Complete');
console.log('='.repeat(80) + '\n');
