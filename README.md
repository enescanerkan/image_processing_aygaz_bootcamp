# Animal Image Classification Project (Hayvan Görüntü Sınıflandırma Projesi)

![sınıflar](https://github.com/user-attachments/assets/24eacf01-2e50-4a92-a47d-7cd2f12af7cc)

This project focuses on classifying animal images using a dataset containing various animal species. (Bu proje, çeşitli hayvan türlerini içeren bir veri kümesi kullanarak hayvan görüntülerini sınıflandırmaya odaklanmaktadır.) The primary aim is to enhance proficiency in image processing and deep learning techniques. (Ana amaç, görüntü işleme ve derin öğrenme tekniklerindeki yetkinliği artırmaktır.)

## Dataset (Veri Kümesi)

The dataset contains images for the following animal species: (Veri kümesi aşağıdaki hayvan türlerine ait görüntüler içermektedir:)

- Leopard (Leopar)
- Dolphin (Yunus)
- Lion (Aslan)
- Fox (Tilki)
- Moose (Geyik)
- Rabbit (Tavşan)
- Horse (At)
- Squirrel (Sincap)
- Bat (Yarasa)
- Gorilla (Goril)
- Antelope (Antilop)

### Key Details: (Ana Detaylar:)

- **Number of Images per Class:** 650 (Her Sınıf için Görüntü Sayısı: 650)
- **Image Resolution:** Resized to 128x128 pixels (Görüntü Çözünürlüğü: 128x128 piksel olarak yeniden boyutlandırıldı)
- **Normalization:** Pixel values normalized to the range [0, 1] to reduce computation time. (Normalizasyon: Hesaplama süresini azaltmak için piksel değerleri [0, 1] aralığına normalleştirildi.)

## Required Libraries and Their Functions (Gerekli Kütüphaneler ve İşlevleri)

- **os, shutil:** File and folder management (e.g., moving, deleting, copying files). (Dosya ve klasör yönetimi, örn. dosyaları taşıma, silme, kopyalama.)
- **OpenCV:** Image processing, resizing, filtering, and other analysis tasks. (Görüntü işleme, yeniden boyutlandırma, filtreleme ve diğer analiz görevleri.)
- **numpy:** Numerical calculations and data manipulation. (Sayısal hesaplamalar ve veri manipülasyonu.)
- **ImageDataGenerator:** Image augmentation and data preparation using Keras. (Keras kullanarak görüntü artırma ve veri hazırlama.)
- **train_test_split:** Splitting the dataset into training and test subsets. (Veri kümesini eğitim ve test alt kümelerine ayırma.)
- **LabelEncoder:** Converting categorical labels into numerical values. (Kategorik etiketleri sayısal değerlere dönüştürme.)

### Keras Layers and Utilities (Keras Katmanları ve Araçları):

- **Model, Dense, Flatten, Dropout, BatchNormalization:** For creating and fine-tuning deep learning models. (Derin öğrenme modelleri oluşturma ve ince ayar yapma için.)
- **Sequential Model:** A simple structure where layers are stacked sequentially. (Katmanların sıralı bir şekilde istiflendiği basit bir yapı.)
- **Functional API:** Allows complex architectures involving branching and merging. (Dallanma ve birleştirme içeren karmaşık mimarilere olanak tanır.)

## Project Workflow (Proje İş Akışı)

### Data Preprocessing (Veri Ön İşleme)

- **Dataset Filtering:** Images for each species are organized into folders with 650 images per class. (Her tür için görüntüler, her sınıfta 650 görüntü olacak şekilde klasörlere düzenlenmiştir.)
- **Splitting Data:** Training and validation sets are created using an 80-20 split. (Eğitim ve doğrulama kümeleri, %80-%20 oranında oluşturulmuştur.)
- **Data Normalization:** Images are resized to 128x128 pixels and normalized to the [0, 1] range. (Görüntüler 128x128 piksel boyutuna yeniden boyutlandırılmış ve [0, 1] aralığına normalleştirilmiştir.)
- **Label Encoding:** Categorical labels are transformed into numerical values using LabelEncoder and one-hot encoding. (Kategorik etiketler, LabelEncoder ve one-hot encoding kullanılarak sayısal değerlere dönüştürülmüştür.)

### Data Augmentation (Veri Artırma)

#### Augmented Images Examples:

![cow](https://github.com/user-attachments/assets/73074e8c-3c44-403c-96b0-bfeb2a580988)
![gorilla](https://github.com/user-attachments/assets/faac7f73-80da-47e9-98ce-42bda103bfdb)

Techniques used: (Kullanılan Teknikler:)

- Rotation (± 30 degrees) (Döndürme (± 30 derece))
- Salt-and-pepper noise (Tuz ve biber gürültüsü)
- Brightness adjustments (Parlaklık ayarlamaları)
- Horizontal flipping (Yatay çevirme)

Augmented images are saved with a prefix (e.g., aug_). (Artırılmış görüntüler bir ön ekle kaydedilir (örn. aug_).)

### Model Architecture (Model Mimarisi)

Layers: (Katmanlar:)

- **Input Layer:** Accepts images with dimensions (128, 128, 3). (Giriş Katmanı: (128, 128, 3) boyutlarında görüntüleri kabul eder.)
- **Convolutional Layers (Conv2D):** Extract features like edges and colors. (Konvolüsyonel Katmanlar (Conv2D): Kenar ve renk gibi özellikleri çıkarır.)
- **Pooling Layers (MaxPooling2D):** Reduce the spatial dimensions of feature maps. (Havuzlama Katmanları (MaxPooling2D): Özellik haritalarının uzaysal boyutlarını azaltır.)
- **Flatten Layer:** Converts 2D outputs to a 1D vector. (Düzleştirme Katmanı: 2D çıktıları 1D bir vektöre dönüştürür.)
- **Dense Layers:** Higher-level feature extraction and classification. (Yoğun Katmanlar: Daha üst düzey özellik çıkarımı ve sınıflandırma.)
- **Output Layer:** Uses softmax activation for multi-class classification. (Çıkış Katmanı: Çok sınıflı sınıflandırma için softmax aktivasyonunu kullanır.)

#### Model Summary: (Model Özeti:)

```python
model_cnn = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3)),
    LeakyReLU(alpha=0.1),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128),
    LeakyReLU(alpha=0.1),
    layers.Dense(len(classes), activation='softmax')
])
```

### Compilation and Training (Derleme ve Eğitim)

- **Optimizer:** RMSprop with a learning rate of 0.001 (Optimize Edici: 0.001 öğrenme oranına sahip RMSprop)
- **Loss Function:** Categorical Crossentropy (Kayıp Fonksiyonu: Kategorik Çapraz Entropi)
- **Metrics:** Accuracy (Metrikler: Doğruluk)

Data augmentation is applied during training to improve generalization and reduce overfitting. (Eğitim sırasında veri artırma, genelleştirmeyi geliştirmek ve aşırı öğrenmeyi azaltmak için uygulanır.)

### Relu and Leaky Relu  Activation Functions 
![relu](https://github.com/user-attachments/assets/b54d6d3f-c806-42f5-af53-de301357c763)

## Visualization of Results (Sonuçların Görselleştirilmesi)

Accuracy and loss graphs are plotted using Matplotlib to analyze the model's performance over epochs. (Doğruluk ve kayıp grafikleri, modelin dönemler boyunca performansını analiz etmek için Matplotlib kullanılarak çizilir.)

## Results (Sonuçlar)

### RMSprop and Leaky Relu Result Graphs:
![Rmsprop](https://github.com/user-attachments/assets/2ae78b45-ef70-4f15-b8de-d88a1701b850)

### Adam and Relu Result Graphs:
![Ekran görüntüsü 2024-12-20 183015](https://github.com/user-attachments/assets/b213bbfa-49ca-42eb-82da-53281d0bbd70)

The model successfully classifies animal images into their respective categories. (Model, hayvan görüntülerini başarıyla ilgili kategorilere sınıflandırır.) Fine-tuning of hyperparameters and additional data augmentation techniques further improve accuracy. (Hiperparametrelerin ince ayarı ve ek veri artırma teknikleri doğruluğu daha da artırır.)

## Conclusion (Sonuç)

This project demonstrates the application of deep learning and image processing techniques in animal classification. (Bu proje, hayvan sınıflandırmada derin öğrenme ve görüntü işleme tekniklerinin uygulanmasını göstermektedir.) Future work could explore incorporating additional animal species, leveraging transfer learning for faster convergence, and experimenting with advanced architectures for improved accuracy. (Gelecek çalışmalar, ek hayvan türlerinin dahil edilmesini, daha hızlı yakınsama için transfer öğreniminin kullanılmasını ve daha yüksek doğruluk için gelişmiş mimarilerle denemeler yapılmasını içerebilir.)

Additionally, during training, two combinations of activation functions and optimizers were explored: ReLU with Adam optimizer and Leaky ReLU with RMSprop optimizer. While the Leaky ReLU and RMSprop combination exhibited slower training, it achieved slightly better results compared to the ReLU and Adam pairing. This highlights the importance of experimenting with different configurations to find the optimal setup for specific tasks.(Ek olarak, eğitim sırasında, aktivasyon fonksiyonları ve optimizörlerin iki kombinasyonu araştırıldı: Adam optimizörlü ReLU ve RMSprop optimizörlü Leaky ReLU. Leaky ReLU ve RMSprop kombinasyonu daha yavaş eğitim sergilerken, ReLU ve Adam eşleşmesine kıyasla biraz daha iyi sonuçlar elde etti. Bu, belirli görevler için optimum kurulumu bulmak için farklı yapılandırmalarla deneme yapmanın önemini vurgular.)

