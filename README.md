# Kredi Kartı Dolandırıcılığı Tespiti Projesi

Streamlit: https://fraud-detection-ml-lhxdmuxma3evc4ekudfvzr.streamlit.app/

## 1) Problem Nedir?
Bizim amacımız, kredi kartı işlemlerinin dolandırıcılık olup olmadığını tahmin etmek.  
- Veri setinde `Class` sütunu var:  
  - 0 = normal işlem  
  - 1 = dolandırıcılık (fraud)  
- Fraud işlemler çok az, bu yüzden modelin onları yakalaması zor.  
- Kullanıcı bir işlem miktarı ve zamanı girince, model dolandırıcılık olasılığını tahmin edecek.

---

## 2) Başlangıç (Baseline) Modelimiz ve Sonucu
- Basit bir Logistic Regression modeli yaptık.  
- Sonuçlar (küçük veri örneği ile):
ROC AUC: 0.9194
PR AUC: 0.6660


- Fraud işlemleri çok iyi yakalayamıyordu, recall çok düşüktü.

---

## 3) Özellik (Feature) Denemelerimiz
- Denediğimiz şeyler:  
  - `LogAmount`: işlem miktarının logunu aldık  
  - `Amt_by_Time`: işlem miktarını zamana böldük  
- Sayısal değerleri ölçekledik (StandardScaler ile).  
- Train seti SMOTE ile çoğalttık (daha fazla fraud örneği ekledik).  
- Sonuç: model dolandırıcılık işlemlerini daha iyi tahmin etmeye başladı.

---

## 4) Veriyi Nasıl Bölüyoruz ve Neden
- Veriyi %80 eğitim, %20 test olarak böldük.  
- Stratified split kullandık, yani sınıf oranlarını koruduk.  
- Sebep: Fraud örnekleri çok az, test setinde kaybolmasın diye.

---

## 5) Final Modelimiz ve Ön İşleme
- Kullandığımız özellikler: `Time`, `LogAmount`, `Amt_by_Time`  
- İşlem sırası:  
  1. Özellikleri oluşturduk  
  2. Sayısal değerleri ölçekledik  
  3. Train setini SMOTE ile çoğalttık  
- Bu sırayla modelimiz hızlı ve doğru çalışıyor.

---

## 6) Final Model vs Baseline
- Baseline modelde fraud recall çok düşüktü (~0.02-0.12)  
- Final modelde fraud recall ~0.76, accuracy ~0.98  
- Yani modelimiz dolandırıcılık işlemlerini çok daha iyi yakalıyor.

---

## 7) Model İş İhtiyaçları ile Uyumlu mu?
- Evet, model dolandırıcılığı hızlı tespit ediyor.  
- Threshold’u 0.01 yaptık, fraud yakalama oranı yüksek.  
- Yanlış pozitif biraz arttı ama iş açısından kabul edilebilir.

---

## 8) Model Canlıya Çıkınca ve İzleme
- Model ve scaler `joblib` ile kaydedildi.  
- Streamlit ile kullanıcıya açabiliriz.  
- Kullanıcı input: işlem miktarı ve zamanı  
- Çıktı: dolandırıcılık ihtimali + sınıf  
- İzleyeceğimiz metrikler:  
  - ROC AUC ve PR AUC  
  - Fraud recall ve precision  
  - Yanlış negatif/pozitif sayıları
