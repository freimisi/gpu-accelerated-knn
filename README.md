# 🧠 GPU-accelerated kNN

> 🔍 Built as part of the "GPU Computing" course @ LUT University  
> 👨‍💻 Contributors: Omer AHMED, Mihály FREI

<p align="center">
  <img src="https://img.shields.io/badge/Made%20With-Python-blue?style=for-the-badge&logo=python&logoColor=green">
  <img src="https://img.shields.io/badge/Environment-Google%20Colab%20(T4%20GPU)-orange?style=for-the-badge&logo=googlecolab&logoColor=orange">
  <img src="https://img.shields.io/badge/Accelerated%20By-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=green">
</p>

## 🖼️ Project Overview 
This project implements a **GPU-accelerated k-Nearest Neighbour** (kNN) classifier, showcasing the speedup that GPU computing (via **CuPy** and **CUDA**) can bring compared to traditional CPU implementations. The kNN logic is written using CuPy's **RawKernel**, giving low-level GPU control for optimal performance.

---

## 🧪 Dataset
The dataset used contains 4000 samples with 7 numerical features and float-type labels.  
After min-max normalization, the labels were cast to integers starting from 0.

**Class distribution:**

| **labels** | **counts** | **proportion** |
|------------|------------|----------------|
| 4.0        | 1746       | 43.650%        |
| 3.0        | 1317       | 32.925%        |
| 5.0        | 664        | 16.600%        |
| 2.0        | 133        |  3.325%        |
| 6.0        | 119        |  2.975%        |
| 1.0        | 18         |  0.450%        |
| 7.0        | 3          |  0.075%        |

---

## 🚅 CUDA Kernel

The CUDA kernel is used to move the following computations to the GPU:
- Distance calculation
- Finding and storing the **k** closest data points
- Majority voting

---

## 📊 Results

| **Device**  | **⏳ Inference Time (avg)** | **✅ Accuracy (avg)** |
|-------------|-----------------------------|------------------------|
| CPU         | 0.1697 s                    | 50.93%                 |
| GPU         | 0.0017 s                    | 50.74%                 |
| **Speedup** | **99.82x**                  |                        |
 
---

## ⛔ Limitations
- **k** capped at 32 due to array size limits in kernel  
- Only Euclidean distance and majority vote supported  
- Designed for demonstration, not full production deployment 

---

## 📁 Project Structure
📦 gpu-accelerated-knn/     
├── GPU_knn.ipynb        
└── MLoGPU_data3_train.csv                 
