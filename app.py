from flask import Flask
from flask_pymongo import PyMongo
from flask_caching import Cache

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/expertsystem_database'
app.secret_key = "sistempakar112023iqbal01"
mongo = PyMongo(app)
data_limit = 10
cache = Cache(app)

# Konfigurasi Flask-Caching
app.config['CACHE_TYPE'] = 'simple'  # Jenis cache sederhana
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # Waktu kedaluwarsa cache dalam detik
app.config['CACHE_KEY_PREFIX'] = 'myapp_'  # Awalan kunci cache

from views import views
app.register_blueprint(views, url_prefix="/views")

if __name__ == "__main__":
    app.run(debug=True)