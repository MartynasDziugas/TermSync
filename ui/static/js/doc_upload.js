/**
 * Doc Upload: parinktų failų sąrašas ir „Remove“ (išvalo atitinkamą file input).
 */
(function () {
    var slots = document.querySelectorAll(".doc-file-slot");
    var list = document.getElementById("doc-upload-staged-list");
    var empty = document.getElementById("doc-upload-staged-empty");
    if (!list || !slots.length) return;

    function rebuild() {
        list.innerHTML = "";
        var any = false;
        slots.forEach(function (inp) {
            if (!(inp instanceof HTMLInputElement) || inp.type !== "file") return;
            if (!inp.files || !inp.files.length) return;
            any = true;
            var f = inp.files[0];
            var label = inp.getAttribute("data-slot-label") || inp.name;
            var li = document.createElement("li");
            li.className = "doc-upload-staged-item";
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
        if (empty) empty.style.display = any ? "none" : "";
    }

    slots.forEach(function (inp) {
        inp.addEventListener("change", rebuild);
    });
    rebuild();
})();
