from flask import render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from app import app, db
from app.models import User, UserFile
import os


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.user_id
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/')
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)
    user_files = UserFile.query.filter_by(user_id=user_id).all()

    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    return render_template('dashboard.html', user=user, user_files=user_files)


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('upload_file'))
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('upload_file'))
        if not allowed_file(file.filename):
            flash('Invalid file type', 'error')
            return redirect(url_for('upload_file'))

        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user.username)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        filename = secure_filename(file.filename)
        file_path = os.path.join(user_folder, filename)

        existing_file = UserFile.query.filter_by(user_id=session['user_id'], file_name=filename).first()
        if existing_file:
            flash('File already exists.', 'error')
            return redirect(url_for('upload_file'))

        file.save(file_path)
        new_file = UserFile(user_id=session['user_id'], file_name=filename, file_path=file_path)
        db.session.add(new_file)
        db.session.commit()
        flash('File uploaded successfully', 'success')
        return redirect(url_for('dashboard'))

    return render_template('upload.html')


@app.route('/delete_file/<int:file_id>')
def delete_file(file_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    file_to_delete = UserFile.query.get(file_id)
    if file_to_delete and file_to_delete.user_id == session['user_id']:
        user = User.query.get(session['user_id'])
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user.username)
        file_path = os.path.join(user_folder, file_to_delete.file_name)

        if os.path.exists(file_path):
            os.remove(file_path)

        db.session.delete(file_to_delete)
        db.session.commit()
        flash('File deleted successfully')
    else:
        flash('Unauthorized to delete this file')
    return redirect(url_for('dashboard'))

@app.route('/uploads/<username>/<filename>')
def uploaded_file(username, filename):
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], username)
    return send_from_directory(user_folder, filename)