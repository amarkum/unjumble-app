CREATE DATABASE IF NOT EXISTS unjumble_app;

USE unjumble_app;

CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL
);

CREATE TABLE user_files (
    file_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
