from app import app, db
from app.models import User, UserFile


def create_tables():
    with app.app_context():
        db.create_all()


if __name__ == "__main__":
    create_tables()
    app.run(debug=True, port=8080)
