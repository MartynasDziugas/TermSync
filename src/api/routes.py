from flask import Blueprint, request, render_template, redirect, url_for
from src.database.connection import get_session, init_db
from src.database.queries import get_all_templates, get_pending_sync_jobs, get_active_version
from src.database.models import MedDRAVersion, SyncJob
from src.services.term_update_service import import_version, detect_changes
from src.services.doc_sync_service import create_sync_job, run_sync_job
from src.services.diff_service import get_diff_summary
from src.services.export_service import export_to_tmx, export_to_docx
from src.services.experiment_service import get_all_experiments
from src.parsers.meddra import MedDRAImporter
from src.parsers.template_importer import TemplateImporter
from config import config

bp = Blueprint("ui", __name__)


@bp.route("/")
def index():
    try:
        with get_session() as session:
            version = get_active_version(session)
            templates = get_all_templates(session)
            jobs = get_pending_sync_jobs(session)
            return render_template("index.html",
                active_version=version.version_number if version else None,
                template_count=len(templates),
                pending_jobs=len(jobs),
            )
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/versions")
def versions():
    try:
        with get_session() as session:
            all_versions = session.query(MedDRAVersion).all()
            return render_template("versions.html", versions=all_versions)
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/versions/import", methods=["POST"])
def import_meddra():
    try:
        version_number = request.form.get("version_number")
        file = request.files.get("file")
        upload_path = config.UPLOAD_FOLDER / file.filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(upload_path)
        importer = MedDRAImporter(upload_path)
        if not importer.validate():
            return render_template("error.html", error_message="Netinkamas failas")
        terms = importer.parse()
        with get_session() as session:
            import_version(session, version_number, terms)
        return redirect(url_for("ui.versions"))
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/versions/diff", methods=["GET", "POST"])
def diff():
    try:
        with get_session() as session:
            all_versions = session.query(MedDRAVersion).all()
            summary = None
            if request.method == "POST":
                old_id = int(request.form.get("old_version"))
                new_id = int(request.form.get("new_version"))
                summary = get_diff_summary(session, old_id, new_id)
            return render_template("diff.html", versions=all_versions, summary=summary)
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/templates")
def templates():
    try:
        with get_session() as session:
            all_templates = get_all_templates(session)
            return render_template("templates.html", templates=all_templates)
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/templates/upload", methods=["POST"])
def upload_template():
    try:
        name = request.form.get("name")
        language = request.form.get("language", "LT")
        file = request.files.get("file")
        upload_path = config.UPLOAD_FOLDER / file.filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(upload_path)
        importer = TemplateImporter(upload_path)
        with get_session() as session:
            importer.import_template(session, name, language)
        return redirect(url_for("ui.templates"))
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/sync")
def sync():
    try:
        with get_session() as session:
            jobs = session.query(SyncJob).all()
            templates = get_all_templates(session)
            return render_template("sync.html", jobs=jobs, templates=templates)
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/sync/create", methods=["POST"])
def create_job():
    try:
        old_template_id = int(request.form.get("old_template_id"))
        new_template_id = int(request.form.get("new_template_id"))
        with get_session() as session:
            create_sync_job(session, old_template_id, new_template_id)
        return redirect(url_for("ui.sync"))
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/sync/run/<int:job_id>", methods=["POST"])
def run_job(job_id: int):
    try:
        with get_session() as session:
            job = run_sync_job(session, job_id)
        return render_template("results.html", job=job)
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/experiments")
def experiments():
    try:
        with get_session() as session:
            all_experiments = get_all_experiments(session)
            return render_template("experiments.html", experiments=all_experiments)
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/export")
def export():
    try:
        with get_session() as session:
            all_versions = session.query(MedDRAVersion).all()
            return render_template("export.html", versions=all_versions)
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/export/tmx", methods=["POST"])
def export_tmx():
    try:
        old_id = int(request.form.get("old_version"))
        new_id = int(request.form.get("new_version"))
        output_path = config.UPLOAD_FOLDER / "export.tmx"
        with get_session() as session:
            export_to_tmx(session, old_id, new_id, output_path)
        return redirect(url_for("ui.export"))
    except Exception as e:
        return render_template("error.html", error_message=str(e))


@bp.route("/export/docx", methods=["POST"])
def export_docx():
    try:
        old_id = int(request.form.get("old_version"))
        new_id = int(request.form.get("new_version"))
        output_path = config.UPLOAD_FOLDER / "export.docx"
        with get_session() as session:
            export_to_docx(session, old_id, new_id, output_path)
        return redirect(url_for("ui.export"))
    except Exception as e:
        return render_template("error.html", error_message=str(e))