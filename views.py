from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session
import pandas as pd 
import numpy as np
import json
import plotly
import plotly.express as px
from app import cache
from app import mongo
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from passlib.hash import pbkdf2_sha256 

views = Blueprint(__name__, "views")

@views.route('/')
def home():
    if 'emailuser' in session:
        return render_template("index.html")
    else:
        return redirect(url_for("views.login"))

@views.route('/admin')
def admin():
    if 'emailuser' in session:

        total_users = mongo.db.dataPengguna.count_documents({})
        total_pengetahuan = mongo.db.basisPengetahuan.count_documents({})
        total_aturan = mongo.db.aturanLogika.count_documents({})
        total_kasus = mongo.db.pertanyaanidentifikasi.count_documents({})

        # Mengambil data dari database
        data_personal = list(mongo.db.datapersonalkorban.find({}, {"namakorban": 1,
                                                             "usiakorban": 1,
                                                              "pendidikankorban": 1,
                                                               "asalkorban": 1}))
        data_inferensi = list(mongo.db.pertanyaanidentifikasi.find({}, {"hasil_inferensi": 1}))

        # Menggabungkan data pengguna dan data pertanyaan menjadi satu list
        data_gabungan = []
        for i in range(min(len(data_personal), len(data_inferensi))):
            data_gabungan.append({
                "nama": data_personal[i].get("namakorban"),
                "usia": data_personal[i].get("usiakorban"),
                "tingkat_pendidikan": data_personal[i].get("pendidikankorban"),
                "asal_provinsi": data_personal[i].get("asalkorban"),
                "hasil_inferensi": data_inferensi[i].get("hasil_inferensi")
                })

        return render_template("admin.html", total_users=total_users, total_kasus=total_kasus, 
                                total_pengetahuan=total_pengetahuan, total_aturan=total_aturan, 
                                data=data_gabungan)
    else:
        return redirect(url_for("views.login"))

@views.route("/aturan")
def aturan():
    data_aturan = list(mongo.db.aturanLogika.find({}, {"Kondisi": 1,
                                                        "Jawaban":1,
                                                        "Indikasi":1}))
    
    return render_template("aturanLogika.html", data_aturan=data_aturan)

@views.route("/basis")
def basis():
    data_pengetahuan = list(mongo.db.basisPengetahuan.find({}, {"pertanyaan": 1,
                                                        "jawaban":1}))
    
    return render_template("basis_pengetahuan.html", data_pengetahuan=data_pengetahuan)

@views.route("/about")
def about():
    return render_template("about.html")

@views.route("/visual")
@cache.cached()
def visual():
    # timeseries data jumlah kasus kekerasan 2016-2022
    df = pd.read_csv('static\data\jumlah_kekerasan_7tahun.csv')
    figure1 = px.line(df,
                      x="Tahun",
                      y="Jumlah Kasus", 
                      title="Tren Jumlah Kasus Kekerasan Terhadap Perempuan (2016-2022)",
                      markers=True)
    figure1.update_traces(textposition="top center")
    figure1.update_layout(width=500,
                          height=400,
                          title_font_size=14)
    graph1JSON = json.dumps(figure1, 
                            cls=plotly.utils.PlotlyJSONEncoder)

    # piechart data jenis kekerasan 2016-2022
    df_pie = pd.read_csv('static\data\jenis_kekerasan_7tahun.csv')
    values = df_pie["Kasus"]
    names = df_pie["Jenis Kekerasan"]
    figure2 = px.pie(df_pie,
                     values=values,
                     names=names,
                     title="Top 5 Jenis Kekerasan Terhadap Perempuan (2016-2022)",
                     color=names)
    figure2.update_layout(width=500,
                          height=400,
                          title_font_size=14)
    graph2JSON = json.dumps(figure2, 
                            cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("visualization.html", graph1JSON = graph1JSON, graph2JSON = graph2JSON)

@views.route("/identify")
def identify():
    return render_template("identify.html")

@views.route("/identify-tanya")
def pertanyaan():
    return render_template('identify-tanya.html')

@views.route("/save-data", methods=['POST'])
def save_data():
    nama_lengkap = request.form['namakorban']
    usia = request.form['usiakorban']
    tingkat_pendidikan = request.form['pendidikankorban']
    asal_provinsi = request.form['asalkorban']

    data = {'namakorban':nama_lengkap,
            'usiakorban':usia,
            'pendidikankorban':tingkat_pendidikan,
            'asalkorban':asal_provinsi}

    mongo.db.datapersonalkorban.insert_one(data)

    return redirect(url_for("views.pertanyaan"))

# pemodelan data
df = pd.read_csv("data_for_modelling.csv")
X = df.drop('target', axis=1)
y = df['target']
model = LogisticRegression()
model.fit(X, y)

basis_pengetahuan = {
    'hubungan_pelaku': [0, 1, 2],
    'tempat_kekerasan': [0, 1, 2, 3],
    'kekerasan_fisik': [0, 1],
    'ancaman_intimidasi': [0, 1],
    'penolakan_emosional': [0, 1],
    'tekanan_seksual': [0, 1],
    'pelecehan_seksual': [0, 1],
    'perlakuan_kasar': [0, 1],
    'cedera_fisik': [0, 1],
    'penghinaan': [0, 1],
    'isolasi': [0, 1],
    'penyalahgunaan_ekonomi': [0, 1],
    'keterbatasan_sumber_daya': [0, 1]
}

# Fungsi untuk melakukan inferensi menggunakan basis pengetahuan dan aturan logika
def inferensi(data_input):
    with open('aturan_logika.json') as file:
        aturan_logika = json.load(file)
        
    for aturan in aturan_logika:
        kondisi = aturan['kondisi']
        match = all(data_input.get(kunci) == nilai for kunci, nilai in kondisi.items())
        if match:
            return aturan['indikasi']
    
    return 'Tidak ditemukan indikasi yang sesuai'

@views.route("/save-tanya", methods=['POST'])
def save_tanya():
    jawaban1 = int(request.form['jawaban1'])
    jawaban2 = int(request.form['jawaban2'])
    jawaban3 = int(request.form['jawaban3'])
    jawaban4 = int(request.form['jawaban4'])
    jawaban5 = int(request.form['jawaban5'])
    jawaban6 = int(request.form['jawaban6'])
    jawaban7 = int(request.form['jawaban7'])
    jawaban8 = int(request.form['jawaban8'])
    jawaban9 = int(request.form['jawaban9'])
    jawaban10 = int(request.form['jawaban10'])
    jawaban11 = int(request.form['jawaban11'])
    jawaban12 = int(request.form['jawaban12'])
    jawaban13 = int(request.form['jawaban13'])

    data_input = {
        'hubungan_pelaku': jawaban1,
        'tempat_kekerasan': jawaban2,
        'kekerasan_fisik': jawaban3,
        'ancaman_intimidasi': jawaban4,
        'penolakan_emosional': jawaban5,
        'tekanan_seksual': jawaban6,
        'pelecehan_seksual': jawaban7,
        'perlakuan_kasar': jawaban8,
        'cedera_fisik': jawaban9,
        'penghinaan': jawaban10,
        'isolasi': jawaban11,
        'penyalahgunaan_ekonomi': jawaban12,
        'keterbatasan_sumber_daya': jawaban13
    }

    # Ubah data input ke dalam format numerik sesuai dengan basis pengetahuan
    data_numerik = {kunci: basis_pengetahuan[kunci].index(data_input[kunci]) for kunci in data_input}

    # Lakukan pemodelan data dengan machine learning menggunakan model Anda
    hasil_pemodelan = model.predict([list(data_numerik.values())])

    # Gunakan hasil pemodelan data sebagai input untuk inferensi
    hasil_inferensi = inferensi(data_numerik)

    # simpan data jawaban ke 
    data = {'jawaban1' : jawaban1,
            'jawaban2' : jawaban2,
            'jawaban3' : jawaban3,
            'jawaban4' : jawaban4,
            'jawaban5' : jawaban5,
            'jawaban6' : jawaban6,
            'jawaban7' : jawaban7,
            'jawaban8' : jawaban8,
            'jawaban9' : jawaban9,
            'jawaban10' : jawaban10,
            'jawaban11' : jawaban11,
            'jawaban12' : jawaban12,
            'jawaban13' : jawaban13,
            'hasil_inferensi' : hasil_inferensi}
    
    data_hasilInferensi={'hasil_inferensi' : hasil_inferensi}

    mongo.db.pertanyaanidentifikasi.insert_one(data)

    return render_template('hasil.html', hasil_inferensi=hasil_inferensi)

@views.route('/hasil', methods=['GET'])
def hasil_identifikasi():
    return render_template('hasil.html', hasil_inferensi=hasil_inferensi, hasil_pemodelan=hasil_pemodelan[0])

@views.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['emailuser']
        password = request.form['passuser']

        # Pengguna --> Cek apakah username dan password cocok di database
        user = mongo.db.dataPengguna.find_one({'emailuser': username})

        if user:
            stored_password = user['passuser']
            # Verifikasi kata sandi yang diinputkan dengan kata sandi yang disimpan dalam database
            if pbkdf2_sha256.verify(password, stored_password):
                session['emailuser'] = username
                return redirect(url_for("views.home"))
            else:
                return 'Username atau password salah!'

        # Admin --> Cek apakah username dan password cocok di database
        admin = mongo.db.dataAdmin.find_one({'emailuser': username})

        if admin:
            stored_password = admin['passuser']
            # Verifikasi kata sandi yang diinputkan dengan kata sandi yang disimpan dalam database
            if password == stored_password:
                session['emailuser'] = username
                return redirect(url_for("views.admin"))
            else:
                return 'Username atau password salah!'
        else:
            return 'Username tidak ditemukan!'
    
    return render_template('login.html')

@views.route('/logout')
def logout():
    session.pop('emailuser', None)
    return redirect(url_for("views.login"))

@views.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['emailuser']
        password = request.form['passuser']

        # Cek apakah username sudah digunakan
        existing_user = mongo.db.dataPengguna.find_one({'emailuser': username})
        if existing_user:
            return 'Username sudah digunakan, silakan coba dengan username lain.'

        # Hashing dan salt password
        hashed_password = pbkdf2_sha256.hash(password)

        # Simpan data pengguna baru ke database
        new_user = {'emailuser': username, 'passuser': hashed_password}
        mongo.db.dataPengguna.insert_one(new_user)
        return redirect(url_for("views.login"))
    
    return render_template('signup.html')