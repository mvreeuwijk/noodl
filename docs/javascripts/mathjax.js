// MathJax configuration for pymdownx.arithmatex (generic mode), as prescribed by the
// mkdocs-material documentation. Arithmatex wraps every formula in a span/div of class
// "arithmatex" delimited by \( \) or \[ \]; in a JavaScript string those backslashes must be
// doubled, otherwise the delimiters become plain "(" ")" "[" "]" and MathJax splits formulas
// at every ordinary parenthesis or bracket.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
