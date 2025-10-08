# 🧠 Mental Illness Prevalence Prediction using Machine Learning

## 🌍 SDG Focus: SDG 3 – Good Health and Well-being
This project uses machine learning to identify **high-risk regions** for mental illnesses such as depression and anxiety, helping policymakers allocate mental health resources effectively.

---

## 📊 Dataset
- **Source:** WHO & Global Mental Health datasets  
- **Rows:** 6420 (countries × years, 1990–2019)  
- **Columns:**  
  - Depression, Anxiety, Schizophrenia, Bipolar, Eating disorders (age-standardized, both sexes)  
  - Metadata: Region, Year, Country Code  

---

## 🤖 Machine Learning Approach
- **Model:** Random Forest Classifier  
- **Type:** Supervised Learning  
- **Target:** `high_risk` → regions with high combined prevalence (> 5%)  
- **Features:** Mental illness prevalence columns  
- **Data Split:** 80% training, 20% testing  

---

## ⚙️ Preprocessing
- Filled missing values with column mean  
- Created target column `high_risk` based on prevalence threshold  

---

## 📈 Results
- **Accuracy:** 1.00  
- **F1 Score:** 1.00  
- **Feature Importance:**  
  - Anxiety disorders: 47.7%  
  - Depressive disorders: 30.6%  
  - Bipolar disorders: 9.1%  
  - Eating disorders: 7.4%  
  - Schizophrenia: 5.1%  

- **Prediction Example:**  
  - Input: Depression 4%, Anxiety 3%, Bipolar 2%, Schizophrenia 1%, Eating 1%  
  - **Output:** High Risk → ❌ No  

---

## 🧩 How to Run
1. Clone the repository:  
```bash
git clone https://github.com/Yolisaq/AI-Machine-Learning.git
