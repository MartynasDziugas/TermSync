from flask import Blueprint, request, jsonify
from src.database.connection import get_session, init_db
from src.database.queries import get_all_templates, get_pending_sync_jobs,
from src.services.term_update_service import import_version, detect_changes
from src.services.doc_sync_service import create_sync_job, run_sync_job
from src.parsers.meddra import MedDRAImporter

bp = Blueprint("api", __name__)

@bp.route("/init", methods=["POST"])
def initialize_db():
    try:
        init_db()
        return jsonify({"status": "DB inicializuota"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/version/active", methods=["GET"])
def active_version():
    try:
        with get_session() as session:
            version = get_active_version(session)
            if not version:
                return jsonify({"error": "Nera aktyvios versijos"}), 404
            return jsonify({"version": version.version_number}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/version/import", methods=["POST"])
def import_meddra():
    try:
        data = request.get_json()
        file_path = data.get("file_path")
        version_number = data.get("version_number")

        importer = MedDRAImporter(file_path)
        if not importer.validate():
            return jsonify({"error": "Netinkamas failas"}), 400

        terms = importer.parse()
        with get_session() as session:
            version = import_version(session, version_number, terms)
            return jsonify({"status": "Importuota", "version_id": version.id}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@bp.route("/version/changes", methods=["POST"])
def version_changes():
    try:
        data = request.get_json()
        old_v = data.get("old_version")
        new_v = data.get("new_version")

        with get_session() as session:
            changes = detect_changes(session, old_v, new_v)
            return jsonify({"changes_count": len(changes)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/templates", methods=["GET"])
def templates():
    try:
        with get_session() as session:
            all_templates = get_all_templates(session)
            return jsonify([{"id": t.id, "name": t.name} for t in all_templates]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/sync/create/<int:template_id>", methods=["POST"])
def create_job(template_id: int):
    try:
        with get_session() as session:
            job = create_sync_job(session, template_id)
            return jsonify({"job_id": job.id, "status": job.status.value}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/sync/run/<int:job_id>", methods=["POST"])
def run_job(job_id: int):
    try:
        with get_session() as session:
            job = run_sync_job(session, job_id)
            return jsonify({
                "job_id": job.id,
                "status": job.status.value,
                "segments_updated": job.segments_updated,
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
