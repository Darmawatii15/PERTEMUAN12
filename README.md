# PERTEMUAN12
BELAJAR FACENET

```
Praktikum12Facenet/
│
├── data/
│   ├── train/
│   │   ├── Aca/
│   │   │   ├── A1.jpeg
│   │   │   └── A2.jpeg
│   │   ├── Darma/
│   │   │   ├── d1.jpeg
│   │   │   └── d2.jpeg
│   │   ├── Elsa/
│   │   │   ├── E1.jpeg
│   │   │   └── E2.jpeg
│   │   ├── Tri/
│   │   │   ├── Tri1.jpeg
│   │   │   └── Tri2.jpeg
│   ├── val/
│   │   ├── Aca/
│   │   │   └── val1.jpeg
│   │   ├── Darma/
│   │   │   └── val1.jpeg
│   │   ├── Elsa/
│   │   │   └── val1.jpeg
│   │   ├── Tri/
│   │   │   └── val1.jpeg
│
├── samples/
│   ├── A1.jpeg
│   ├── d1.jpeg
│   └── E1.jpeg
│
├── venv/
│   └── ... (your virtual environment files)
│
├── build_embeddings.py
├── convert_npz_to_npy.py
├── evaluate.py
├── facenet_knn.joblib
├── predict_knn.py
├── svm_model.pkl
├── train_classifier.py
├── train_knn.py
├── utils_facenet.py
├── verify_cli.py
├── verify_pair.py
├── webcam_knn.py
├── X_train.npy
└── y_train.npy

```
# ANALISIS

## build_embeddings.py
Kode build_embeddings.py berfungsi untuk membangun dataset embedding wajah dari kumpulan gambar yang tersimpan di folder data/train. Program dimulai dengan menelusuri setiap subfolder di dalam direktori tersebut, di mana setiap subfolder dianggap sebagai satu identitas atau satu orang. Untuk setiap orang, skrip membaca seluruh file gambar yang ada di dalam foldernya, kemudian memanggil fungsi embed_from_path() dari modul utils_facenet untuk menghasilkan embedding wajah. Embedding ini adalah representasi numerik hasil ekstraksi fitur menggunakan model FaceNet. Jika sebuah gambar tidak mengandung wajah yang dapat dideteksi, maka gambar tersebut dilewati. Embedding yang berhasil diproses disimpan ke dalam list embeddings, sementara label atau nama foldernya disimpan di list labels. Setelah semua gambar selesai diproses, list tersebut dikonversi menjadi array NumPy dan disimpan dalam file embeddings.npz. File ini nantinya akan digunakan pada proses pelatihan atau pengenalan wajah sehingga sistem dapat membedakan identitas berdasarkan embedding yang telah dibuat.
## convert_npz_to_npy.py
Kode convert_npz_to_npy.py berfungsi untuk memisahkan kembali data embedding wajah dan label yang sebelumnya disimpan dalam satu file embeddings.npz. File .npz tersebut dimuat menggunakan numpy.load(), lalu dua array di dalamnya—yakni embeddings sebagai fitur (X) dan labels sebagai target (y)—diambil dari struktur file tersebut. Setelah dipisahkan, masing-masing array disimpan ulang dalam format .npy melalui np.save(), yaitu sebagai X_train.npy dan y_train.npy. Proses ini memudahkan penggunaan data pada tahap pelatihan model karena format .npy lebih sederhana dan cepat diakses. Terakhir, program menampilkan informasi jumlah data dan bentuk (shape) dari array embedding untuk memastikan bahwa pemisahan dan penyimpanan telah dilakukan dengan benar.
## evaluate.py
Kode evaluate.py digunakan untuk menguji performa model klasifikasi wajah berbasis SVM yang telah dilatih sebelumnya. Proses evaluasi dimulai dengan memuat model svm_model.pkl dan menelusuri seluruh folder di dalam direktori validasi data/val, di mana setiap folder mewakili satu identitas. Untuk setiap gambar dalam folder, skrip memanggil fungsi embed_from_path() untuk mengambil embedding wajah menggunakan FaceNet. Jika wajah tidak terdeteksi, gambar dilewati agar tidak memengaruhi perhitungan akurasi. Embedding yang berhasil diekstrak kemudian diprediksi menggunakan model SVM, dan hasil prediksi dibandingkan dengan nama folder sebagai label yang benar. Setiap prediksi yang tepat meningkatkan jumlah correct, sementara seluruh gambar yang tervalidasi menambah nilai total. Setelah seluruh data diuji, program mencetak jumlah sampel, jumlah prediksi benar, dan menghitung akurasi akhir model. Evaluasi ini memberikan gambaran seberapa baik model mampu mengenali identitas wajah berdasarkan embedding yang dihasilkan.
## predict_knn.py
Kode predict_knn.py berfungsi untuk melakukan prediksi identitas wajah pada satu gambar menggunakan model KNN yang telah dilatih sebelumnya. Program dimulai dengan memeriksa apakah pengguna memberikan argumen berupa path gambar saat menjalankan perintah di terminal. Jika tidak ada argumen, skrip menampilkan cara pemakaian yang benar dan menghentikan program. Setelah itu, model KNN dimuat dari file facenet_knn.joblib, lalu gambar yang diberikan diproses menggunakan fungsi embed_from_path() untuk menghasilkan embedding wajah. Apabila wajah tidak terdeteksi, program menghentikan eksekusi untuk menghindari prediksi yang tidak valid. Jika embedding tersedia, model melakukan prediksi identitas menggunakan model.predict() serta menghitung probabilitas setiap kelas dengan model.predict_proba(). Hasil akhirnya berupa label kelas yang diprediksi beserta probabilitasnya, sehingga pengguna dapat melihat baik hasil prediksi maupun tingkat keyakinan model. Skrip ini sangat berguna untuk pengujian cepat terhadap satu gambar tanpa perlu melalui proses evaluasi penuh.
## train_classifier.py
Kode train_classifier.py digunakan untuk melatih model klasifikasi wajah berbasis SVM (Support Vector Machine) menggunakan embedding wajah yang sebelumnya telah diekstraksi dan disimpan dalam file embeddings.npz. Program memulai proses dengan memuat file tersebut dan mengambil dua array utama, yaitu X sebagai embedding berdimensi 512 dan y sebagai label identitas masing-masing embedding. Setelah memastikan bentuk dan isi data, skrip membuat model klasifikasi menggunakan SVC dengan kernel linear, yang umum digunakan untuk embedding FaceNet karena mampu memisahkan ruang fitur secara efektif. Parameter probability=True diaktifkan agar model dapat menghasilkan probabilitas prediksi pada tahap inferensi. Selanjutnya, model dilatih menggunakan data embedding dan label melalui clf.fit(X, y). Setelah proses pelatihan selesai, model disimpan dalam file svm_model.pkl menggunakan joblib.dump(), sehingga dapat digunakan kembali tanpa perlu melakukan pelatihan ulang. Dengan demikian, skrip ini merupakan tahap inti dalam pipeline pengenalan wajah—mengubah embedding wajah menjadi model yang mampu mengklasifikasikan identitas dengan akurasi tinggi.
## train_knn.py
Kode train_knn.py digunakan untuk melatih model klasifikasi wajah berbasis algoritma k-Nearest Neighbors (KNN) menggunakan embedding wajah yang sebelumnya telah disimpan dalam file X_train.npy dan y_train.npy. Proses dimulai dengan memuat kedua file tersebut: X berisi embedding fitur wajah, sedangkan y berisi label identitas. Selanjutnya, kode membangun sebuah pipeline yang terdiri dari dua tahap, yaitu StandardScaler dan KNeighborsClassifier. Tahap scaling dilakukan untuk menyamakan skala setiap fitur agar perhitungan jarak pada algoritma KNN tidak bias terhadap dimensi yang memiliki nilai lebih besar. Setelah data distandarisasi, model KNN dengan n_neighbors=3 dan metrik jarak Euclidean digunakan sebagai classifier untuk mengelompokkan embedding berdasarkan kedekatan jarak antar titik di ruang fitur. Model kemudian dilatih menggunakan clf.fit(X, y) hingga mampu mengenali pola kedekatan antar embedding. Setelah pelatihan selesai, seluruh pipeline—termasuk scaler dan model KNN—disimpan ke dalam file facenet_knn.joblib menggunakan joblib.dump(), sehingga dapat langsung digunakan untuk prediksi tanpa perlu melakukan preprocessing ulang. Secara keseluruhan, skrip ini membangun model pengenalan wajah berdasarkan kemiripan jarak antar embedding, yang merupakan pendekatan sederhana namun efektif untuk sistem face recognition.
## utils_facenet.py
File utils_facenet.py berisi kumpulan fungsi pendukung yang digunakan untuk membaca gambar, mendeteksi wajah, melakukan alignment, serta menghasilkan embedding wajah menggunakan FaceNet. Pada bagian awal, skrip menentukan perangkat komputasi yang tersedia—GPU atau CPU—sebelum memuat dua komponen penting: MTCNN sebagai face detector sekaligus face aligner, dan InceptionResnetV1 sebagai model embedder yang menghasilkan representasi fitur wajah berukuran 512 dimensi. Fungsi read_img_bgr() membaca gambar dalam format BGR menggunakan OpenCV, sementara bgr_to_pil() mengonversi format BGR ke RGB untuk memudahkan pemrosesan oleh PIL dan MTCNN. Fungsi face_align() menggunakan MTCNN untuk mendeteksi serta mengekstrak wajah yang sudah ter-align dalam ukuran 160×160 piksel; jika wajah tidak ditemukan, fungsi mengembalikan None. Fungsi embed_face_tensor() kemudian mengambil wajah yang sudah di-align, mengubahnya menjadi batch tensor, memprosesnya dengan model embedder, dan menghasilkan embedding 512-dim dalam bentuk NumPy array. Semua operasi dilakukan dalam mode torch.no_grad() agar lebih efisien dan tidak menyimpan gradien. Fungsi embed_from_path() menyediakan alur lengkap dari membaca gambar hingga menghasilkan embedding, sehingga dapat digunakan langsung dalam proses pelatihan maupun prediksi. Selain itu, tersedia fungsi cosine_similarity() untuk menghitung kemiripan antar dua embedding berdasarkan cosinus. Secara keseluruhan, modul ini menjadi fondasi utama dari pipeline face recognition, menyediakan seluruh proses praproses hingga ekstraksi embedding wajah.
## verify_cli.py
Kode verify_cli.py digunakan sebagai alat verifikasi wajah berbasis command line, yang memungkinkan pengguna membandingkan dua gambar wajah untuk menentukan apakah keduanya merupakan orang yang sama. Program menerima dua argumen berupa path gambar (img1 dan img2) serta opsi threshold kemiripan (--th) dengan nilai default 0.85. Setelah argumen diproses, skrip menghasilkan embedding untuk masing-masing gambar menggunakan fungsi embed_from_path() dari modul utils_facenet. Jika salah satu gambar tidak mengandung wajah yang terdeteksi, program langsung menampilkan pesan kesalahan. Jika kedua embedding berhasil dibuat, skrip menghitung nilai kemiripan menggunakan cosine_similarity(), yang mengukur seberapa dekat arah vektor embedding satu dengan lainnya. Nilai similarity kemudian dicetak, dan program mengambil keputusan dengan membandingkannya terhadap nilai threshold: jika similarity di atas atau sama dengan threshold, gambar dianggap sebagai "MATCH" (wajah sama), sedangkan jika lebih rendah maka dianggap "NO MATCH". Dengan demikian, skrip ini berfungsi sebagai alat verifikasi wajah sederhana namun efektif tanpa memerlukan model klasifikasi tambahan.
## verify_pair.py
Kode verify_pair.py digunakan untuk melakukan verifikasi wajah secara sederhana dengan membandingkan dua gambar tertentu yang telah ditentukan langsung di dalam skrip. Dua variabel, img1 dan img2, diisi dengan path gambar yang ingin diuji. Kemudian masing-masing gambar diproses menggunakan embed_from_path() untuk menghasilkan embedding wajah. Jika salah satu gambar tidak memiliki wajah yang terdeteksi, skrip menampilkan pesan kesalahan dan menghentikan proses verifikasi. Jika kedua embedding berhasil dihasilkan, nilai kemiripan dihitung menggunakan cosine_similarity(), yang mengukur seberapa dekat arah kedua vektor embedding dalam ruang fitur. Nilai similarity tersebut kemudian dibandingkan dengan threshold yang ditentukan (0.85), dan skrip menampilkan apakah kedua gambar dianggap sebagai wajah yang sama atau tidak. Dengan demikian, kode ini merupakan versi lebih sederhana dari verifikasi CLI, cocok untuk pengujian cepat tanpa memerlukan input dari command line.
## webcam_knn.py
Kode ini digunakan untuk melakukan pengenalan wajah secara realtime menggunakan webcam dengan memanfaatkan model FaceNet untuk ekstraksi fitur dan KNN untuk klasifikasi. Pertama, program memuat model MTCNN sebagai pendeteksi dan aligner wajah, serta model KNN yang telah dilatih sebelumnya, lengkap dengan daftar label identitasnya. Setelah webcam berhasil dibuka, setiap frame yang masuk diproses dengan MTCNN untuk mendeteksi dan mengekstrak wajah yang sudah ter-align. Jika wajah berhasil ditemukan, embedding wajah dihitung melalui fungsi embed_from_image(), kemudian embedding tersebut diklasifikasikan oleh model KNN. Probabilitas prediksi dihitung, dan jika nilai tertinggi melebihi threshold 0.55 maka wajah dikenali sebagai identitas tertentu; jika tidak, label "Unknown" ditampilkan. Selanjutnya, kotak deteksi wajah dan hasil prediksi ditampilkan di atas frame video menggunakan OpenCV. Proses ini berlangsung terus-menerus sampai pengguna menekan tombol ‘q’ untuk keluar. Secara keseluruhan, skrip ini memadukan deteksi wajah, ekstraksi fitur, dan klasifikasi dalam satu alur yang efisien sehingga mampu melakukan pengenalan wajah secara langsung melalui kamera.
