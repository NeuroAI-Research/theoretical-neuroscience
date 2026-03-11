document.addEventListener("DOMContentLoaded", function () {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false }, // This catches the MkDocs output
      { left: "\\[", right: "\\]", display: true }   // This catches display-mode output
    ],
    throwOnError: false
  });
});