from app import db
from werkzeug.security import generate_password_hash, check_password_hash


class User(db.Model):
    """
    User model representing the users table in the database.
    """
    __tablename__ = 'users'

    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    # Relationship to link users to their files
    files = db.relationship('UserFile', backref='user', lazy='dynamic')

    def set_password(self, password):
        """
        Create hashed password.
        """
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """
        Check if provided password matches the stored hashed password.
        """
        return check_password_hash(self.password_hash, password)


class UserFile(db.Model):
    """
    UserFile model representing the user_files table in the database.
    """
    __tablename__ = 'user_files'

    file_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f'<UserFile {self.file_name}>'
