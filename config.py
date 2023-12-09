import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_very_secret_key_you_should_change'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'mysql+pymysql://root:adminadmin@localhost/unjumble_app'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = '/opt/uploads/assets'
