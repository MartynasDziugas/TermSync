/**
 * Krovimo indikatorius: dešinėje vertikaliai pildomas pastelinės rožinės juosta.
 * Progresas nuolat juda (eksponentinis artėjimas), ne „užstringa“ ties fiksuotu %.
 */
(function () {
    var side = document.getElementById("global-loading-side");
    var fill = side && side.querySelector(".global-loading-side-fill");
    if (!side || !fill) return;

    var depth = 0;
    var raf = 0;
    var progress = 0;
    var lastT = 0;
    var finishing = false;
    var finishToken = 0;

    function cancelRaf() {
        if (raf) {
            window.cancelAnimationFrame(raf);
            raf = 0;
        }
    }

    /** Kol aktyvu: nuolat juda link 100% (be fiksuotos viršutinės ribos iki stop()). */
    function tick(t) {
        if (!side.classList.contains("is-active") || finishing) return;
        if (!lastT) lastT = t;
        var dt = Math.min(100, t - lastT);
        lastT = t;
        progress += (100 - progress) * (dt / 1000) * 1.22;
        fill.style.height = progress + "%";
        raf = window.requestAnimationFrame(tick);
    }

    function start() {
        depth += 1;
        if (depth > 1) return;
        finishing = false;
        finishToken += 1;
        cancelRaf();
        progress = 0;
        lastT = 0;
        fill.style.height = "0%";
        side.classList.add("is-active");
        raf = window.requestAnimationFrame(tick);
    }

    function stop() {
        depth = Math.max(0, depth - 1);
        if (depth > 0) return;
        cancelRaf();
        finishing = true;
        var token = ++finishToken;
        var startP = progress;
        var t0 = performance.now();
        var duration = 240;

        function finishTween(now) {
            if (token !== finishToken) return;
            if (!side.classList.contains("is-active")) return;
            var u = Math.min(1, (now - t0) / duration);
            var eased = 1 - (1 - u) * (1 - u);
            var h = startP + (100 - startP) * eased;
            fill.style.height = h + "%";
            if (u < 1) {
                raf = window.requestAnimationFrame(finishTween);
            } else {
                window.setTimeout(function () {
                    if (token !== finishToken) return;
                    side.classList.remove("is-active");
                    fill.style.height = "0%";
                    progress = 0;
                    finishing = false;
                    lastT = 0;
                }, 160);
            }
        }
        raf = window.requestAnimationFrame(finishTween);
    }

    document.addEventListener(
        "submit",
        function (e) {
            var form = e.target;
            if (!(form instanceof HTMLFormElement)) return;
            if (form.getAttribute("data-no-global-loading") === "1") return;
            start();
        },
        true
    );

    var origFetch = window.fetch;
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
        finishToken += 1;
        finishing = false;
        cancelRaf();
        side.classList.remove("is-active");
        fill.style.height = "0%";
        progress = 0;
        lastT = 0;
    });

    window.TermSyncLoading = { start: start, stop: stop };
})();
