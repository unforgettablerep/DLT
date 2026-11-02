// Render an SVG to PNG (raster preview) and PDF (vector) with the pre-installed Chromium.
// usage: node render.js in.svg out.png out.pdf [scale]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
(async () => {
  const [svgPath, pngPath, pdfPath, scaleArg] = process.argv.slice(2);
  const svg = fs.readFileSync(svgPath, 'utf8');
  const m = svg.match(/width="(\d+)" height="(\d+)"/);
  const w = +m[1], h = +m[2];
  const scale = +(scaleArg || 2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: scale });
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0;background:#fff}</style></head><body>${svg}</body></html>`;
  await page.setContent(html, { waitUntil: 'load' });
  await page.screenshot({ path: pngPath, clip: { x: 0, y: 0, width: w, height: h } });
  if (pdfPath) {
    await page.pdf({ path: pdfPath, width: `${w}px`, height: `${h}px`, printBackground: true, pageRanges: '1' });
  }
  await browser.close();
  console.log('rendered', path.basename(pngPath), pdfPath ? path.basename(pdfPath) : '');
})();
