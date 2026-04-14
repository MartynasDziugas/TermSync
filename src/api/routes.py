import json
import re
import uuid
from io import StringIO
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from config import config
from src.parsers.docx_parser import DocxParser
from src.services.csv_glossary_service import decode_glossary_file_bytes, parse_glossary_csv
from src.services.data_maintenance_service import (
    delete_all_standard_train_uploads,
    delete_standard_train_upload_file,
    delete_translator_check_upload,
    list_standard_train_upload_pairs,
    purge_all_glossary_csv_disk,
    purge_glossary_disk_files_named,
    purge_review_sessions,
    purge_standard_train_uploads_matching_meta_pair,
    purge_training_data,
    purge_translator_check_uploads,
    purge_translator_uploads_having_docx_basenames,
)
from src.services.persist_service import (
    count_all_glossary_rows,
    count_glossary_rows,
    delete_all_glossary_batches,
    delete_glossary_batch,
    delete_glossary_batches_by_filename,
    delete_review_session,
    get_glossary_batch,
    get_review_session,
    get_training_run_by_job_id,
    list_glossary_batches,
    list_glossary_rows,
    list_review_history,
    list_training_history,
    persist_glossary_csv,
    persist_review_session,
    review_rows_from_session,
)
from src.services.review_export_service import review_rows_to_csv_bytes
from src.services.standard_train_service import (
    get_latest_train_ui_snapshot,
    get_train_job_api_fallback,
    load_latest_bundle,
)
from src.services.translator_resume_service import (
    list_glossary_disk_uploads,
    list_resumable_translator_uploads,
    list_translator_disk_inventory,
    resolve_paths_for_regenerate,
    write_translator_session_meta,
)
from src.services.translator_review_service import run_translator_review

bp = Blueprint("ui", __name__)


def _import_glossary_from_upload(file_storage, *, has_header: bool) -> tuple[int, int]:
    """Įrašo vienkartinį CSV žodyną į DB; grąžina (batch.id, eilučių skaičius)."""
    if not file_storage or not file_storage.filename:
        raise ValueError("Pasirinkite .csv žodyno failą.")
    if not file_storage.filename.lower().endswith(".csv"):
        raise ValueError("Žodynas turi būti .csv formatas.")
    fname = secure_filename(file_storage.filename)
    delete_glossary_batches_by_filename(fname)
    purge_glossary_disk_files_named(fname)
    uid = uuid.uuid4().hex[:10]
    dest_dir = config.UPLOAD_FOLDER / "glossary_csv" / uid
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / fname
    file_storage.save(dest)
    raw = dest.read_bytes()
    text, enc_used = decode_glossary_file_bytes(raw)
    rows, warns = parse_glossary_csv(StringIO(text), has_header=has_header)
    for w in warns:
        flash(w, "info")
    if enc_used not in ("utf-8", "utf-8-sig"):
        flash(
            f"CSV koduotė: {enc_used}. Jei lietuviškos raidės neteisingos, naudokite „CSV UTF-8“.",
            "info",
        )
    batch_id = persist_glossary_csv(filename=fname, rows=rows)
    return batch_id, len(rows)


def _redirect_after_translator_or_review_delete():
    """Iš Doc Upload /check puslapio POST — atgal į /check; iš Istorijos — į /history."""
    ref = (request.headers.get("Referer") or "").lower()
    if "/check" in ref and "/history" not in ref:
        return redirect(url_for("ui.check"))
    return redirect(url_for("ui.history"))


TRAIN_JOB_RE = re.compile(r"^[a-f0-9]{16}$")
REVIEW_SESSION_RE = re.compile(r"^[a-f0-9]{32}$")
TRANSLATOR_UPLOAD_UID_RE = re.compile(r"^[a-f0-9]{12}$")


def _train_page_kwargs() -> dict[str, object]:
    """train.html: paskutinio mokymo snapshot + įkeltos poros diske."""
    snap = get_latest_train_ui_snapshot()
    if snap is not None:
        jid = snap.get("job_id")
        if isinstance(jid, str) and TRAIN_JOB_RE.match(jid):
            snap = {
                **snap,
                "set_active_model_url": url_for("ui.train_job_set_active_model", job_id=jid),
            }
    return {
        "train_snapshot": snap,
        "upload_pairs": list_standard_train_upload_pairs(),
    }


def _train_worker(
    job_id: str,
    left_path: Path,
    right_path: Path,
    src_name: str,
    tgt_name: str,
) -> None:
    from src.services.standard_train_service import run_standard_train_job

    run_standard_train_job(
        left_path,
        right_path,
        job_id,
        display_src_name=src_name,
        display_tgt_name=tgt_name,
        hyperparams=None,
    )


def _train_json_hyperparams() -> tuple[dict, int]:
    """POST /train su application/json: tas pats pipeline kaip fone, sinchroniai + metrics."""
    from src.services.pair_train_job_store import ensure_job_slot, get_job
    from src.services.standard_train_service import run_standard_train_job

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return {"ok": False, "error": "JSON body must be an object"}, 400
    job_id = (payload.get("job_id") or "").strip()
    if not TRAIN_JOB_RE.match(job_id):
        return {
            "ok": False,
            "error": (
                "Missing or invalid job_id (16 lowercase hex). "
                "Upload both standard .docx on the Train page first — the redirect URL contains the job id."
            ),
        }, 400
    base = config.UPLOAD_FOLDER / "standard_train" / job_id
    left_path = base / "standard_src.docx"
    right_path = base / "standard_tgt.docx"
    if not left_path.is_file() or not right_path.is_file():
        return {
            "ok": False,
            "error": f"No uploaded documents for job_id={job_id!r} (expected standard_src.docx and standard_tgt.docx).",
        }, 400
    for p in (left_path, right_path):
        if not DocxParser(p).validate():
            return {"ok": False, "error": f"Invalid .docx: {p.name}"}, 400

    hp = {k: payload[k] for k in ("bert", "svm", "mlp") if k in payload}
    ensure_job_slot(job_id)
    result = run_standard_train_job(
        left_path,
        right_path,
        job_id,
        display_src_name=left_path.name,
        display_tgt_name=right_path.name,
        hyperparams=hp if hp else None,
    )
    if result is None:
        row = get_job(job_id) or {}
        return {"ok": False, "error": row.get("error", "Training failed.")}, 400
    return {"ok": True, "job_id": job_id, "metrics": result}, 200


@bp.route("/")
def index():
    latest = load_latest_bundle()
    has_model = latest is not None
    run_id = latest[1] if latest else None
    return render_template("index.html", has_model=has_model, run_id=run_id)


@bp.route("/history")
def history():
    training_runs = list_training_history()
    review_sessions = list_review_history()
    glossary_batches = list_glossary_batches()
    return render_template(
        "history.html",
        training_runs=training_runs,
        review_sessions=review_sessions,
        glossary_batches=glossary_batches,
        translator_disk_inventory=list_translator_disk_inventory(),
        glossary_disk_uploads=list_glossary_disk_uploads(),
    )


@bp.route("/history/maintenance", methods=["POST"])
def history_maintenance():
    action = request.form.get("action", "")
    redirect_frag = "#duomenu-valymas"
    try:
        if action == "purge_training":
            stats = purge_training_data()
            flash(
                f"Ištrinta standarto mokymų: {stats['training_runs']}; "
                f"pašalinta artefaktų aplankų: {stats['artifact_job_dirs']}.",
                "success",
            )
        elif action == "purge_review_sessions":
            also = request.form.get("also_delete_files") == "on"
            stats = purge_review_sessions(delete_files=also)
            flash(
                f"Pašalinta peržiūros sesijų: {stats['sessions']}; "
                f"pašalinta failų iš disko: {stats['files']}.",
                "success",
            )
        elif action == "purge_review_uploads":
            stats = purge_translator_check_uploads()
            flash(
                f"Išvalyta translator_check įrašų: {stats['entries']}. "
                "DB sesijos nebuvo trintos (lentelė vis dar rodo senas nuorodas).",
                "success",
            )
        elif action == "purge_glossary_csv_disk":
            stats = purge_all_glossary_csv_disk()
            flash(
                f"Pašalinta įrašų iš glossary_csv: {stats['entries']}.",
                "success",
            )
            redirect_frag = "#history-disk-glossary"
        elif action == "purge_glossary_db_batches":
            n = delete_all_glossary_batches()
            flash(f"Pašalintos DB žodyno partijos: {n}.", "success")
            redirect_frag = "#history-glossary-db"
        else:
            flash("Nežinomas veiksmas.", "error")
    except Exception as e:
        flash(str(e), "error")
    return redirect(url_for("ui.history") + redirect_frag)


@bp.route("/history/translator-upload/<uid>/delete", methods=["POST"])
def translator_upload_delete(uid: str):
    if not TRANSLATOR_UPLOAD_UID_RE.match(uid):
        abort(404)
    try:
        stats = delete_translator_check_upload(uid)
        msg = "Įkėlimo aplankas pašalintas iš disko."
        if stats["sessions"]:
            msg += f" Pašalinta susietų DB sesijų: {stats['sessions']}."
        flash(msg, "success")
    except ValueError as e:
        flash(str(e), "error")
    return _redirect_after_translator_or_review_delete()


@bp.route("/history/review-session/<sid>/delete", methods=["POST"])
def review_session_delete(sid: str):
    if not REVIEW_SESSION_RE.match(sid):
        abort(404)
    also_files = request.form.get("delete_files") == "on"
    if not delete_review_session(sid, delete_files=also_files):
        abort(404)
    flash(
        "Sesija pašalinta iš DB."
        + (
            " Pašalinti ir susieti .docx failai."
            if also_files
            else " Vertėjo .docx liko diske — puslapyje „Vertėjo tikrinimas“ galite „Generuoti naują sesiją su jau įkeltais failais“."
        ),
        "success",
    )
    return _redirect_after_translator_or_review_delete()


@bp.route("/train/hyperparams")
def hyperparam_dashboard():
    return render_template("hyperparam_dashboard.html")


@bp.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "GET":
        return render_template(
            "train.html",
            error_message=None,
            quick_epochs=config.QUICK_MLP_EPOCHS,
            **_train_page_kwargs(),
        )
    if request.method == "POST" and request.is_json:
        body, code = _train_json_hyperparams()
        return jsonify(body), code
    try:
        from src.services.pair_train_job_store import create_job, run_in_thread

        left_f = request.files.get("standard_src")
        right_f = request.files.get("standard_tgt")
        if not left_f or not left_f.filename or not right_f or not right_f.filename:
            raise ValueError("Įkelkite abu standarto .docx failus (source ir target).")
        src_nm = secure_filename(left_f.filename)
        tgt_nm = secure_filename(right_f.filename)
        purge_standard_train_uploads_matching_meta_pair(src_nm, tgt_nm)
        job_id = create_job()
        job_dir = config.UPLOAD_FOLDER / "standard_train" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        left_path = job_dir / "standard_src.docx"
        right_path = job_dir / "standard_tgt.docx"
        left_f.save(left_path)
        right_f.save(right_path)
        for p in (left_path, right_path):
            dp = DocxParser(p)
            if not dp.validate():
                raise ValueError(f"Netinkamas .docx: {p.name}")
        (job_dir / "upload_meta.json").write_text(
            json.dumps(
                {"standard_src_name": src_nm, "standard_tgt_name": tgt_nm},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        run_in_thread(_train_worker, (job_id, left_path, right_path, src_nm, tgt_nm))
        return redirect(url_for("ui.train_job", job_id=job_id))
    except Exception as e:
        return render_template(
            "train.html",
            error_message=str(e),
            quick_epochs=config.QUICK_MLP_EPOCHS,
            **_train_page_kwargs(),
        )


@bp.route("/train/uploads/delete-all", methods=["POST"])
def train_uploads_delete_all():
    removed = delete_all_standard_train_uploads()
    from src.services.pair_train_job_store import remove_job

    for jid in removed:
        remove_job(jid)
    flash(f"Pašalinti standarto įkėlimai diske: {len(removed)}.", "success")
    return redirect(url_for("ui.train"))


@bp.route("/train/uploads/<job_id>/delete-one", methods=["POST"])
def train_upload_delete_one(job_id: str):
    if not TRAIN_JOB_RE.match(job_id):
        abort(404)
    role = (request.form.get("role") or "").strip().lower()
    if role not in ("src", "tgt"):
        flash("Netinkamas failo tipas.", "error")
        return redirect(url_for("ui.train"))
    outcome = delete_standard_train_upload_file(job_id, role)
    if outcome == "folder":
        from src.services.pair_train_job_store import remove_job

        remove_job(job_id)
    return redirect(url_for("ui.train"))


@bp.route("/train/job/<job_id>")
def train_job(job_id: str):
    if not TRAIN_JOB_RE.match(job_id):
        abort(404)
    return render_template("train_job.html", job_id=job_id)


@bp.route("/train/api/<job_id>")
def train_api(job_id: str):
    if not TRAIN_JOB_RE.match(job_id):
        abort(404)
    from src.services.pair_train_job_store import get_job

    row = get_job(job_id)
    if row is None:
        row = get_train_job_api_fallback(job_id)
        if row is None:
            abort(404)
    else:
        row = dict(row)

    if row.get("status") == "done":
        tr = get_training_run_by_job_id(job_id)
        if tr is not None:
            row["active_model_type"] = (tr.active_model_type or "svm").lower()
        else:
            res = row.get("result")
            if isinstance(res, dict):
                row["active_model_type"] = str(res.get("active_model_type") or "svm").lower()
            else:
                row["active_model_type"] = "svm"
    return jsonify(row)


@bp.route("/train/job/<job_id>/active-model", methods=["POST"])
def train_job_set_active_model(job_id: str):
    if not TRAIN_JOB_RE.match(job_id):
        abort(404)
    from src.services.active_model_service import set_active_model

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Tikėtasi JSON."}), 400
    mt = (payload.get("model_type") or "").strip().lower()
    try:
        saved = set_active_model(mt, job_id=job_id)
        return jsonify({"ok": True, "active_model_type": saved}), 200
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@bp.route("/check", methods=["GET", "POST"])
def check():
    has_model = load_latest_bundle() is not None
    if request.method == "GET":
        return render_template(
            "doc_upload.html",
            has_model=has_model,
            error_message=None,
            rows=None,
            run_id=None,
            session_public_id=None,
            session_saved=False,
            glossary_count=count_all_glossary_rows(),
            resumable_uploads=list_resumable_translator_uploads(),
            review_session=None,
        )
    try:
        if not has_model:
            raise ValueError("Pirmiausia paleiskite mokymą su standarto poromis.")
        source_lang = (request.form.get("source_lang") or "en").strip()[:16]
        target_lang = (request.form.get("target_lang") or "lt").strip()[:16]
        gl_f = request.files.get("glossary_csv")
        if gl_f and gl_f.filename:
            has_gl_header = request.form.get("glossary_has_header", "1") == "1"
            batch_id, n_gl = _import_glossary_from_upload(gl_f, has_header=has_gl_header)
            flash(f"Žodynas įrašytas į DB: {n_gl} eil. (partija #{batch_id}).", "success")
        glossary_imported = bool(gl_f and gl_f.filename)
        ts_f = request.files.get("translator_src")
        tt_f = request.files.get("translator_tgt")
        if not ts_f or not ts_f.filename or not tt_f or not tt_f.filename:
            raise ValueError("Įkelkite vertėjo source ir target .docx failus (žemiau sąraše).")
        ts_disk = secure_filename(ts_f.filename or "src.docx")
        tt_disk = secure_filename(tt_f.filename or "tgt.docx")
        purge_translator_uploads_having_docx_basenames(ts_disk, tt_disk)
        uid = uuid.uuid4().hex[:12]
        base = config.UPLOAD_FOLDER / "translator_check" / uid
        base.mkdir(parents=True, exist_ok=True)
        ts_path = base / ts_disk
        tt_path = base / tt_disk
        ts_f.save(ts_path)
        tt_f.save(tt_path)
        for p in (ts_path, tt_path):
            if not DocxParser(p).validate():
                raise ValueError(f"Netinkamas .docx: {p.name}")
        write_translator_session_meta(
            base,
            translator_src_display=secure_filename(ts_f.filename),
            translator_tgt_display=secure_filename(tt_f.filename),
            src_disk_name=ts_path.name,
            tgt_disk_name=tt_path.name,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        use_glossary = glossary_imported or (request.form.get("use_glossary") == "on")
        rows, run_id = run_translator_review(
            ts_path, tt_path, use_glossary=use_glossary
        )
        public_id = uuid.uuid4().hex
        gloss_upload_name = (
            secure_filename(gl_f.filename) if glossary_imported and gl_f and gl_f.filename else None
        )
        gloss_from_db = (not glossary_imported) and (request.form.get("use_glossary") == "on")
        persist_review_session(
            public_id=public_id,
            training_run_job_id=run_id,
            translator_src_name=secure_filename(ts_f.filename),
            translator_tgt_name=secure_filename(tt_f.filename),
            translator_src_path=ts_path,
            translator_tgt_path=tt_path,
            rows=rows,
            glossary_upload_filename=gloss_upload_name,
            glossary_from_db=gloss_from_db,
        )
        return redirect(url_for("ui.check_session", sid=public_id))
    except Exception as e:
        return render_template(
            "doc_upload.html",
            has_model=has_model,
            error_message=str(e),
            rows=None,
            run_id=None,
            session_public_id=None,
            session_saved=False,
            glossary_count=count_all_glossary_rows(),
            resumable_uploads=list_resumable_translator_uploads(),
            review_session=None,
        )


@bp.route("/check/regenerate", methods=["POST"])
def check_regenerate():
    has_model = load_latest_bundle() is not None
    uid = (request.form.get("upload_uid") or "").strip()
    if not re.match(r"^[a-f0-9]{12}$", uid):
        flash("Netinkamas įkėlimo ID.", "error")
        return redirect(url_for("ui.check"))
    if not has_model:
        flash("Nėra aktyvaus standarto modelio.", "error")
        return redirect(url_for("ui.check"))
    resolved = resolve_paths_for_regenerate(uid)
    if resolved is None:
        flash("Įkėlimas nerastas arba trūksta session_meta.json / .docx.", "error")
        return redirect(url_for("ui.check"))
    ts_path, tt_path, src_nm, tgt_nm = resolved
    try:
        for p in (ts_path, tt_path):
            if not DocxParser(p).validate():
                raise ValueError(f"Netinkamas .docx: {p.name}")
        use_glossary = request.form.get("use_glossary") == "on"
        rows, run_id = run_translator_review(
            ts_path, tt_path, use_glossary=use_glossary
        )
        public_id = uuid.uuid4().hex
        persist_review_session(
            public_id=public_id,
            training_run_job_id=run_id,
            translator_src_name=src_nm,
            translator_tgt_name=tgt_nm,
            translator_src_path=ts_path,
            translator_tgt_path=tt_path,
            rows=rows,
            glossary_upload_filename=None,
            glossary_from_db=use_glossary,
        )
        return redirect(url_for("ui.check_session", sid=public_id))
    except Exception as e:
        flash(str(e), "error")
        return redirect(url_for("ui.check"))


@bp.route("/check/session/<sid>")
def check_session(sid: str):
    if not REVIEW_SESSION_RE.match(sid):
        abort(404)
    rec = get_review_session(sid)
    if rec is None:
        abort(404)
    rows = review_rows_from_session(rec)
    return render_template(
        "doc_upload.html",
        has_model=True,
        error_message=None,
        rows=rows,
        run_id=rec.training_run_job_id,
        session_public_id=rec.public_id,
        session_saved=True,
        glossary_count=count_all_glossary_rows(),
        resumable_uploads=list_resumable_translator_uploads(),
        review_session=rec,
    )


@bp.route("/check/session/<sid>/export.csv")
def check_session_export(sid: str):
    if not REVIEW_SESSION_RE.match(sid):
        abort(404)
    rec = get_review_session(sid)
    if rec is None:
        abort(404)
    rows = review_rows_from_session(rec)
    data = review_rows_to_csv_bytes(rows)
    name = f"review_{sid}.csv"
    return Response(
        data,
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
        },
    )


@bp.route("/glossary", methods=["GET", "POST"])
def glossary():
    batches = list_glossary_batches()
    if request.method == "GET":
        return render_template("glossary.html", batches=batches, error_message=None)
    try:
        up = request.files.get("csv_file")
        if not up or not up.filename:
            raise ValueError("Pasirinkite CSV failą.")
        has_header = request.form.get("has_header", "1") == "1"
        batch_id, nrows = _import_glossary_from_upload(up, has_header=has_header)
        flash(f"Įrašyta į DB: {nrows} eilučių (partija #{batch_id}).", "success")
        return redirect(url_for("ui.glossary_batch", batch_id=batch_id))
    except Exception as e:
        return render_template(
            "glossary.html",
            batches=list_glossary_batches(),
            error_message=str(e),
        )


@bp.route("/glossary/batch/<int:batch_id>/delete", methods=["POST"])
def glossary_batch_delete(batch_id: int):
    if not delete_glossary_batch(batch_id):
        abort(404)
    flash(
        "CSV glossoriaus partija pašalinta iš DB. Galite įkelti naują failą. "
        "(Senas failas gali likti diske po data/uploads/glossary_csv — neprivaloma trinti.)",
        "success",
    )
    return redirect(url_for("ui.glossary"))


@bp.route("/glossary/batch/<int:batch_id>")
def glossary_batch(batch_id: int):
    batch = get_glossary_batch(batch_id)
    if batch is None:
        abort(404)
    rows = list_glossary_rows(batch_id, limit=500)
    total = count_glossary_rows(batch_id)
    return render_template(
        "glossary_batch.html",
        batch=batch,
        rows=rows,
        total=total,
    )
