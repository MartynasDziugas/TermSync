from flask import flask
from src.api.routes import bp
from src.database.connection import init_db
from config import config

def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(config)

    app.register_blueprint(bp, url_prefix="/api")

    with app.app_context():
        init_db()

    return app

if __name__== "__main__":
    app = create_app()
    app.run(debug=config.DEBUG)
    