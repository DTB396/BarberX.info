const fs = require('fs');
const path = require('path');

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function walk(dir) {
  fs.readdirSync(dir).forEach(file => {
    const full = path.join(dir, file);
    if (fs.lstatSync(full).isDirectory()) return walk(full);

    const newName = slugify(path.parse(file).name) + path.extname(file);
    if (newName !== file) {
      fs.renameSync(full, path.join(dir, newName));
    }
  });
}

walk('_cases');
