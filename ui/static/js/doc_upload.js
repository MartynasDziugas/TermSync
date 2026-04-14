/**
 * Doc Upload: Remove šalia slot'o, Remove all — vertėjui / žodynui.
 * .doc-upload-staged-server eilutės: Remove pašalina tik DOM (rodinį), ne DB.
 */
(function () {
    var slots = document.querySelectorAll(".doc-file-slot");
    var listTrans = document.getElementById("doc-upload-staged-list-trans");
    var listGloss = document.getElementById("doc-upload-staged-list-gloss");
    var emptyTrans = document.getElementById("doc-upload-staged-empty-trans");
    var emptyGloss = document.getElementById("doc-upload-staged-empty-gloss");
    var btnAllTrans = document.getElementById("doc-remove-all-trans");
    var btnAllGloss = document.getElementById("doc-remove-all-gloss");
    if (!listTrans || !listGloss || !slots.length) return;

    function updateEmptyStates() {
        if (emptyTrans) {
            emptyTrans.style.display = listTrans.querySelector(".doc-upload-staged-item")
                ? "none"
                : "";
        }
        if (emptyGloss) {
            emptyGloss.style.display = listGloss.querySelector(".doc-upload-staged-item")
                ? "none"
                : "";
        }
    }

    function removeServerRows(ul) {
        ul.querySelectorAll("li.doc-upload-staged-server").forEach(function (n) {
            n.remove();
        });
    }

    function clearGroup(group) {
        slots.forEach(function (inp) {
            if (!(inp instanceof HTMLInputElement) || inp.type !== "file") return;
            if ((inp.getAttribute("data-slot-group") || "translator") !== group) return;
            inp.value = "";
            inp.dispatchEvent(new Event("change", { bubbles: true }));
        });
    }

    function rebuild() {
        listTrans.querySelectorAll(".doc-upload-staged-client").forEach(function (n) {
            n.remove();
        });
        listGloss.querySelectorAll(".doc-upload-staged-client").forEach(function (n) {
            n.remove();
        });
        slots.forEach(function (inp) {
            if (!(inp instanceof HTMLInputElement) || inp.type !== "file") return;
            if (!inp.files || !inp.files.length) return;
            var group = inp.getAttribute("data-slot-group") || "translator";
            var list = group === "glossary" ? listGloss : listTrans;
            var f = inp.files[0];
            var label = inp.getAttribute("data-slot-label") || inp.name;
            var li = document.createElement("li");
            li.className = "doc-upload-staged-item doc-upload-staged-client";
            var name = document.createElement("span");
            name.className = "doc-upload-staged-name";
            name.textContent = label + ": " + f.name;
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "doc-upload-remove";
            btn.textContent = "Remove";
            btn.setAttribute("aria-label", "Remove " + f.name);
            btn.addEventListener("click", function () {
                inp.value = "";
                inp.dispatchEvent(new Event("change", { bubbles: true }));
            });
            li.appendChild(name);
            li.appendChild(btn);
            list.appendChild(li);
        });
        updateEmptyStates();
    }

    function onListRemoveClick(ev, ul) {
        var t = ev.target;
        if (!(t instanceof Element)) return;
        var btn = t.closest("button.doc-upload-remove");
        if (!btn || btn.classList.contains("doc-upload-remove-all")) return;
        var li = btn.closest("li.doc-upload-staged-server");
        if (!li || li.parentElement !== ul) return;
        li.remove();
        updateEmptyStates();
    }

    listTrans.addEventListener("click", function (e) {
        onListRemoveClick(e, listTrans);
    });
    listGloss.addEventListener("click", function (e) {
        onListRemoveClick(e, listGloss);
    });

    if (btnAllTrans) {
        btnAllTrans.addEventListener("click", function () {
            clearGroup("translator");
            removeServerRows(listTrans);
            updateEmptyStates();
        });
    }
    if (btnAllGloss) {
        btnAllGloss.addEventListener("click", function () {
            clearGroup("glossary");
            removeServerRows(listGloss);
            updateEmptyStates();
        });
    }

    slots.forEach(function (inp) {
        inp.addEventListener("change", rebuild);
    });
    rebuild();
})();
