/**
 * Viršutinė krovimo linija: formų siuntimas, fetch, vidinės nuorodos.
 */
(function () {
    const bar = document.getElementById("global-loading-bar");
    if (!bar) return;

    let depth = 0;

    function start() {
        depth += 1;
        bar.classList.add("is-active");
    }

    function stop() {
        depth = Math.max(0, depth - 1);
        if (depth === 0) {
            bar.classList.remove("is-active");
        }
    }

    document.addEventListener(
        "submit",
        function (e) {
            const form = e.target;
            if (!(form instanceof HTMLFormElement)) return;
            if (form.getAttribute("data-no-global-loading") === "1") return;
            start();
        },
        true
    );

    const origFetch = window.fetch;
    window.fetch = function () {
        start();
        var p = origFetch.apply(this, arguments);
        if (p && typeof p.finally === "function") {
            return p.finally(function () {
                stop();
            });
        }
        return Promise.resolve(p)
            .then(function (r) {
                stop();
                return r;
            })
            .catch(function (err) {
                stop();
                throw err;
            });
    };

    document.addEventListener("click", function (e) {
        if (e.defaultPrevented || typeof e.target.closest !== "function") return;
        if (e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
        var a = e.target.closest && e.target.closest("a[href]");
        if (!a || a.target === "_blank" || a.hasAttribute("download")) return;
        var href = a.getAttribute("href");
        if (!href || href.charAt(0) === "#" || href.indexOf("javascript:") === 0) return;
        try {
            var u = new URL(a.href, window.location.href);
            if (u.origin !== window.location.origin) return;
        } catch (_) {
            return;
        }
        start();
    });

    window.addEventListener("pageshow", function () {
        depth = 0;
        bar.classList.remove("is-active");
    });

    window.TermSyncLoading = { start: start, stop: stop };
})();
