// 세 페이지(/chat, /eval, /viewer)가 공유하는 상단 탭 메뉴. <div id="nav"></div> 자리에 그려진다.
(function () {
  const TABS = [
    ["/chat", "질문하기"],
    ["/eval", "평가 결과"],
    ["/viewer", "데이터 뷰어"],
  ];
  const style = document.createElement("style");
  style.textContent = `
    #nav { position: sticky; top: 0; z-index: 10; background: var(--card, #fff); border-bottom: 1px solid var(--border, #e3e5ea);
           display: flex; align-items: center; gap: 4px; padding: 0 20px; flex-shrink: 0; }
    #nav .brand { font-weight: 700; margin-right: 20px; padding: 14px 0; white-space: nowrap; }
    #nav a.tab { color: var(--muted, #6b7280); text-decoration: none; padding: 14px 14px; font-size: 14px; font-weight: 500;
                 border-bottom: 2px solid transparent; margin-bottom: -1px; white-space: nowrap; }
    #nav a.tab:hover { color: var(--text, #1d2330); }
    #nav a.tab.on { color: var(--accent, #4f46e5); border-bottom-color: var(--accent, #4f46e5); font-weight: 600; }
    #nav .spacer { flex: 1; }
    #nav a.docs { color: var(--muted, #6b7280); font-size: 12px; text-decoration: none; }
    @media (max-width: 520px) { #nav { padding: 0 8px; } #nav .brand { display: none; } #nav a.tab { padding: 14px 10px; } }`;
  document.head.appendChild(style);

  const here = location.pathname.replace(/\/$/, "");
  const el = document.getElementById("nav");
  if (!el) return;
  el.innerHTML = `<span class="brand">AI 기본법 RAG</span>` +
    TABS.map(([href, label]) => `<a class="tab ${here === href ? "on" : ""}" href="${href}">${label}</a>`).join("") +
    `<span class="spacer"></span><a class="docs" href="/docs">API 문서</a>`;
})();
