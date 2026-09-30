// Collect player-prop markets from a VBET Ukraine game page, in your own Chrome
// (Claude in Chrome runs this with its javascript tool on the open game page).
// Read-only: it clicks the market-category tabs and scrolls the list, nothing else.
//
// The market list is virtualised (only on-screen groups exist in the DOM), so
// each tab is scrolled top to bottom and every group is captured as it renders.
// Output (window.__vbAll), one market per line, read by wnba/scraping/saved_lines.py:
//   #R7;Name;1.08;Name;2.75                ladder "7 or more rebounds" (P/R/A/T = pts/reb/ast/threes)
//   Name (Team) Очки, Асисти та Підбирання;Більше;Менше;30.5;1.92;1.78   over/under
// To keep it short, names become @0, @1 ... with a first line "#N;name0;name1;..",
// "most points/rebounds/..." markets are dropped, and ladder rungs priced
// outside 1.20-6.00 are dropped.
// A tool call returns ~1,000 characters, so read it back with
//   window.__vbAll.slice(0, 950), window.__vbAll.slice(950, 1900), ...
// and save the joined text as data/vbet_<US date>_<AWAY>@<HOME>.txt

window.__vbCollect = async function () {
  const items = {};
  const grab = () => document.querySelectorAll('[data-item-index]').forEach(e => {
    const t = e.innerText.trim();
    if (t) items[e.dataset.itemIndex] = t.replace(/\n+/g, ' | ');
  });
  let sc = document.querySelector('[data-item-index]');
  while (sc && !(sc.scrollHeight > sc.clientHeight + 5 && /(auto|scroll)/.test(getComputedStyle(sc).overflowY))) sc = sc.parentElement;
  sc = sc || document.scrollingElement;
  sc.scrollTop = 0; await new Promise(r => setTimeout(r, 400)); grab();
  for (let i = 0; i < 80; i++) {
    const before = sc.scrollTop;
    sc.scrollTop += 400;
    await new Promise(r => setTimeout(r, 300));
    grab();
    if (sc.scrollTop === before) break;
  }
  sc.scrollTop = 0;
  return Object.keys(items).sort((a, b) => a - b).map(k => items[k]).join('\n');
};

(async () => {
  const tabs = {
    pts: 'Гравці', reb: 'Підбирання гравця', ast: 'Асисти гравців',
    fg3m: 'Триочкові результативні кидки гравців', pra: 'Очки, Асисти та Підбирання Гравців',
  };
  window.__vb = {};
  for (const [k, label] of Object.entries(tabs)) {
    const el = [...document.querySelectorAll('div,span,a,li')]
      .find(e => e.childElementCount <= 1 && e.innerText && e.innerText.trim().startsWith(label));
    if (!el) { window.__vb[k] = ''; continue; }
    el.click();
    await new Promise(r => setTimeout(r, 1500));
    window.__vb[k] = await window.__vbCollect();
  }
  window.__vbAll = Object.values(window.__vb).join('\n')
    .replace(/Гравець набере (\d+) або більше очок/g, '#P$1')
    .replace(/Гравець (?:зробить|здійснить) (\d+) або більше [Пп]ідбирань/g, '#R$1')
    .replace(/Гравець (?:зробить|здійснить) (\d+) або більше [Аа]ссистів/g, '#A$1')
    .replace(/Гравець (?:зробить|здійснить) (\d+) або більше результативних триочкових кидків/g, '#T$1')
    .replace(/ \| /g, ';');
  const names = [], idx = {};
  const lines = window.__vbAll.split('\n').filter(l => l.startsWith('#') || l.includes('Більше')).map(l => {
    if (!l.startsWith('#')) return l;
    const t = l.split(';'), out = [t[0]];
    for (let i = 1; i + 1 < t.length; i += 2) {
      const o = parseFloat(t[i + 1]);
      if (isNaN(o) || !isNaN(parseFloat(t[i]))) { i -= 1; continue; }   // malformed pair
      if (o < 1.2 || o > 6.0) continue;
      if (!(t[i] in idx)) { idx[t[i]] = names.length; names.push(t[i]); }
      out.push('@' + idx[t[i]], t[i + 1]);
    }
    return out.length > 1 ? out.join(';') : null;
  }).filter(Boolean);
  window.__vbAll = ['#N;' + names.join(';'), ...lines].join('\n');
  return window.__vbAll.length;
})();
