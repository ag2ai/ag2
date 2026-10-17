/*
 * Theme the AG2 Code Assistant widget (embed.js) on the documentation site.
 *
 * The embed exposes only `data-accent` for theming; its panel body is hardcoded
 * #fff inside an OPEN shadow DOM, so we inject override stylesheets there once the
 * widget mounts. This relies on the vendor's internal class names and may need
 * updating if the embed markup changes (it fails safe — if the classes go away the
 * panel just falls back to its defaults, nothing breaks).
 *
 * Two overrides, applied differently because the docs site has a light/dark toggle:
 *   - Resize  : always applied (theme-neutral).
 *   - Tint    : light-lavender surfaces, applied ONLY in the docs light theme.
 *               Material's dark scheme ("slate") keeps the widget's own dark-safe
 *               defaults, so the panel stays readable in dark mode. We watch the
 *               <body data-md-color-scheme> attribute and toggle the tint live when
 *               the reader flips the palette.
 */
(function themeAgentWidget() {
    const RESIZE = `
        .agentos-panel { width: 800px !important; max-width: calc(100vw - 40px) !important; height: 660px !important; }
    `;
    const TINT = `
        .agentos-panel { background: #efecf5 !important; }
        .agentos-list { background: #e7e3ef !important; }
        .agentos-bubble-agent { background: #f6f4fa !important; border-color: #dcd6e8 !important; }
        .agentos-composer { background: #efecf5 !important; border-top-color: #dcd6e8 !important; }
        .agentos-footer { background: #efecf5 !important; }
        .agentos-input, textarea { background: #fbfafd !important; }
    `;

    // Material's dark palette is scheme "slate"; anything else (e.g. "default") is light.
    const isLightTheme = () => document.body.getAttribute('data-md-color-scheme') !== 'slate';

    function ensureStyle(sr, id, css) {
        let el = sr.getElementById(id);
        if (!el) {
            el = document.createElement('style');
            el.id = id;
            el.textContent = css;
            sr.appendChild(el);
        }
        return el;
    }

    function apply(sr) {
        ensureStyle(sr, 'ag2-widget-resize', RESIZE); // always on
        if (isLightTheme()) {
            ensureStyle(sr, 'ag2-widget-tint', TINT);
        } else {
            const tint = sr.getElementById('ag2-widget-tint');
            if (tint) tint.remove();
        }
    }

    let tries = 0;
    const timer = setInterval(() => {
        const host = document.getElementById('agentos-widget-root');
        const sr = host && host.shadowRoot;
        if (sr) {
            clearInterval(timer);
            apply(sr);
            // Re-toggle the tint whenever the reader flips the docs light/dark palette.
            new MutationObserver(() => apply(sr)).observe(document.body, {
                attributes: true,
                attributeFilter: ['data-md-color-scheme'],
            });
        } else if (++tries > 100) {
            clearInterval(timer); // give up after ~20s
        }
    }, 200);
})();
