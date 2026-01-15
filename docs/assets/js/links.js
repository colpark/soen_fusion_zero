(function(){
  try {
    var host = window.location.hostname;
    // Only run on GitHub Pages hosts
    var isPages = host.endsWith('github.io') || host.endsWith('pages.github.io');
    if (!isPages) return;

    var anchors = document.querySelectorAll('a[href$=".md"]');
    anchors.forEach(function(a){
      var href = a.getAttribute('href');
      // ignore external links
      if (/^https?:\/\//i.test(href)) return;
      a.setAttribute('href', href.replace(/\.md(#|$)/, '.html$1'));
    });
  } catch (e) {
    console && console.warn && console.warn('Link rewrite failed', e);
  }
})();
