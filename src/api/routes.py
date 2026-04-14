import re
import uuid
from io import StringIO
from pathlib import Path

from flask import (
    Blueprint,
    Response,
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
from src.services.csv_glossary_service import parse_glossary_csv
from src.services.data_maintenance_service import (
    delete_translator_check_upload,
    purge_review_sessions,
    purge_training_data,
    purge_translator_check_uploads,
)
from src.services.persist_service import (
    count_all_glossary_rows,
    count_glossary_rows,
    delete_review_session,
    get_glossary_batch,
    get_review_session,
    list_glossary_batches,
    list_glossary_rows,
    list_review_history,
    list_training_history,
    persist_glossary_csv,
    persist_review_session,
    review_rows_from_session,
)
from src.services.review_export_service import review_rows_to_csv_bytes
from src.services.standard_train_service import load_latest_bundle
from src.services.translator_resume_service import (
    list_resumable_translator_uploads,
    list_translator_disk_inventory,
    resolve_paths_for_regenerate,
    write_translator_session_meta,
)
from src.services.translator_review_service import run_translator_review

bp = Blueprint("ui", __name__)

TRAIN_JOB_RE = re.compile(r"^[a-f0-9]{16}$")
REVIEW_SESSION_RE = re.compile(r"^[a-f0-9]{32}$")
TRANSLATOR_UPLOAD_UID_RE = re.compile(r"^[a-f0-9]{12}$")


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
    )


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
    )


@bp.route("/history/maintenance", methods=["POST"])
def history_maintenance():
    action = request.form.get("action", "")
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
        else:
            flash("Nežinomas veiksmas.", "error")
    except Exception as e:
        flash(str(e), "error")
    return redirect(url_for("ui.history") + "#duomenu-valymas")


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
    return redirect(url_for("ui.history"))


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
    return redirect(url_for("ui.history"))


@bp.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "GET":
        return render_template(
            "train.html",
            error_message=None,
            quick_epochs=config.QUICK_MLP_EPOCHS,
        )
    try:
        from src.services.pair_train_job_store import create_job, run_in_thread

        left_f = request.files.get("standard_src")
        right_f = request.files.get("standard_tgt")
        if not left_f or not left_f.filename or not right_f or not right_f.filename:
            raise ValueError("Įkelkite abu standarto .docx failus (source ir target).")
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
        src_nm = secure_filename(left_f.filename)
        tgt_nm = secure_filename(right_f.filename)
        run_in_thread(_train_worker, (job_id, left_path, right_path, src_nm, tgt_nm))
        return redirect(url_for("ui.train_job", job_id=job_id))
    except Exception as e:
        return render_template(
            "train.html",
            error_message=str(e),
            quick_epochs=config.QUICK_MLP_EPOCHS,
        )


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
        abort(404)
    return jsonify(row)


@bp.route("/check", methods=["GET", "POST"])
def check():
    has_model = load_latest_bundle() is not None
    if request.method == "GET":
        return render_template(
            "check.html",
            has_model=has_model,
            error_message=None,
            rows=None,
            run_id=None,
            session_public_id=None,
            session_saved=False,
            glossary_count=count_all_glossary_rows(),
            resumable_uploads=list_resumable_translator_uploads(),
        )
    try:
        if not has_model:
            raise ValueError("Pirmiausia paleiskite mokymą su standarto poromis.")
        ts_f = request.files.get("translator_src")
        tt_f = request.files.get("translator_tgt")
        if not ts_f or not ts_f.filename or not tt_f or not tt_f.filename:
            raise ValueError("Įkelkite vertėjo source ir target .docx failus.")
        uid = uuid.uuid4().hex[:12]
        base = config.UPLOAD_FOLDER / "translator_check" / uid
        base.mkdir(parents=True, exist_ok=True)
        ts_path = base / secure_filename(ts_f.filename or "src.docx")
        tt_path = base / secure_filename(tt_f.filename or "tgt.docx")
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
        )
        use_glossary = request.form.get("use_glossary") == "on"
        rows, run_id = run_translator_review(
            ts_path, tt_path, use_glossary=use_glossary
        )
        public_id = uuid.uuid4().hex
        persist_review_session(
            public_id=public_id,
            training_run_job_id=run_id,
            translator_src_name=secure_filename(ts_f.filename),
            translator_tgt_name=secure_filename(tt_f.filename),
            translator_src_path=ts_path,
            translator_tgt_path=tt_path,
            rows=rows,
        )
        return redirect(url_for("ui.check_session", sid=public_id))
    except Exception as e:
        return render_template(
            "check.html",
            has_model=has_model,
            error_message=str(e),
            rows=None,
            run_id=None,
            session_public_id=None,
            session_saved=False,
            glossary_count=count_all_glossary_rows(),
            resumable_uploads=list_resumable_translator_uploads(),
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
        "check.html",
        has_model=True,
        error_message=None,
        rows=rows,
        run_id=rec.training_run_job_id,
        session_public_id=rec.public_id,
        session_saved=True,
        glossary_count=count_all_glossary_rows(),
        resumable_uploads=list_resumable_translator_uploads(),
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
        if not up.filename.lower().endswith(".csv"):
            raise ValueError("Leidžiamas tik .csv formatas.")
        has_header = request.form.get("has_header", "1") == "1"
        uid = uuid.uuid4().hex[:10]
        dest_dir = config.UPLOAD_FOLDER / "glossary_csv" / uid
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / secure_filename(up.filename)
        up.save(dest)
        text = dest.read_text(encoding="utf-8-sig")
        rows, warns = parse_glossary_csv(StringIO(text), has_header=has_header)
        fname = secure_filename(up.filename)
        batch_id = persist_glossary_csv(filename=fname, rows=rows)
        for w in warns:
            flash(w, "info")
        flash(f"Įrašyta į DB: {len(rows)} eilučių (partija #{batch_id}).", "success")
        return redirect(url_for("ui.glossary_batch", batch_id=batch_id))
    except Exception as e:
        return render_template(
            "glossary.html",
            batches=list_glossary_batches(),
            error_message=str(e),
        )


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
