from flask import Flask

from config import config
from src.api.routes import bp
from src.database.connection import init_db


def create_app() -> Flask:
    app = Flask(__name__, template_folder="ui/templates", static_folder="ui/static")
    app.config.from_object(config)
    app.register_blueprint(bp, url_prefix="")
    with app.app_context():
        init_db()
    return app

if __name__== "__main__":
    app = create_app()
    app.run(debug=config.DEBUG)
    