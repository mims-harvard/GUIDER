const copyButton = document.querySelector('#copy-citation');
const citation = document.querySelector('#bibtex');

copyButton?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyButton.textContent = 'Copied';
    window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1800);
  } catch {
    copyButton.textContent = 'Select text';
  }
});
