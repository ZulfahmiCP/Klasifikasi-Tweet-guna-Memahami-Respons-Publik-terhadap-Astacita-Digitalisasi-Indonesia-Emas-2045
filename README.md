# ANFORCOM DDSC 2025: Astacita Tweet Classification 🇮🇩

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Project Overview
Repository ini berisi solusi *end-to-end* untuk kompetisi **ANFORCOM Diponegoro Data Science Competition (DDSC) 2025**.

**Tugas Utama:** Mengklasifikasikan opini publik (Tweet/X) terkait topik digitalisasi ke dalam 8 Pilar "Astacita" (Visi Indonesia Emas 2045)[cite: 71, 76].
**Tantangan:** Menggunakan *Small Language Model* (< 500M parameter) dengan batasan maksimal 3 kali submission.

> **Context for Recruiters:** Proyek ini dikerjakan sebagai simulasi *production-ready NLP pipeline*, dengan fokus pada efisiensi model (IndoBERT) dan teknik *Data Cleaning* yang agresif untuk menangani *noisy text* Bahasa Indonesia.

## Alignment with GoTo / Sahabat-AI Mission
Solusi ini dirancang dengan prinsip-prinsip yang selaras dengan inisiatif LLM Bahasa Indonesia (Sahabat-AI):

1.  **Data-Centric AI Approach:** Fokus utama bukan hanya pada arsitektur model, tetapi pada pipeline **Data Cleaning & Normalisasi** yang robust (Regex) untuk menangani slang, singkatan, dan noise data. [cite_start]Ini relevan dengan tanggung jawab intern dalam *data collection, validation, and labeling*.
2.  **Computational Efficiency:** Menggunakan `indobert-base-p2` (110M params) alih-alih LLM raksasa, membuktikan bahwa performa tinggi bisa dicapai dengan *resource* terbatas melalui *fine-tuning* yang tepat.
3.  **Local Context Awareness:** Model dioptimalkan untuk memahami nuansa linguistik lokal (Bahasa Indonesia informal), kunci utama dalam pengembangan LLM nasional.

## 🛠️ Methodology & Pipeline

### 1. Advanced Preprocessing (The "Cleaning" Engine)
Data media sosial sangat kotor. Saya menerapkan pipeline pembersihan bertingkat:
- **Noise Removal:** Menghapus URL, Usernames (@), dan Hashtag yang tidak relevan.
- **Emoji Handling:** Mengonversi emoji bendera menjadi teks (e.g., 🇮🇩 -> "indonesia") untuk mempertahankan konteks semantik.
- **Slang Normalization:** Menggunakan *heuristic rules* untuk menstandarisasi singkatan umum (e.g., "yg" -> "yang", "gpp" -> "tidak apa-apa").

### 2. Model Architecture
- **Base Model:** `indobert-base-p2` (Pre-trained by IndoLEM).
- **Training Strategy:**
  - *Mixed Precision Training (FP16)* untuk efisiensi memori pada GPU T4.
  - *Gradient Accumulation* untuk menjaga stabilitas batch size.
  - *Cross-Entropy Loss* dengan *Class Weighting* untuk menangani ketidakseimbangan data antar pilar Astacita.

### 3. Validation Strategy ("The Sniper Approach")
Karena batasan submission yang ketat (Max 3), validasi lokal adalah kunci:
- **Stratified K-Fold (k=5):** Memastikan distribusi label yang seimbang di setiap fold.
- **Metric Monitoring:** Memantau Weighted F1-Score dan Accuracy secara real-time.
- **Out-of-Fold (OOF) Analysis:** Menganalisis prediksi gabungan dari 5 model untuk estimasi akurasi leaderboard yang presisi.

---

## 📂 Repository Structure
