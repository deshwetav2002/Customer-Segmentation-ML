📊 Customer Segmentation System

Using Rule-Based and Machine Learning Approaches

Final Year Project – Semester 7

📌 Project Overview

This project implements a Customer Segmentation System that groups customers into meaningful segments using both:
Rule-Based Segmentation (feature engineering & business logic)
Machine Learning–Based Segmentation (K-Means clustering)
The system is designed with a clean, modular architecture, where:
Streamlit is used only for the user interface
All preprocessing, feature engineering, and ML logic are handled in backend modules
File-based storage is used to keep the system simple and extensible
This approach ensures clarity, interpretability, and future scalability, making the project suitable for academic evaluation and real-world extension.

🎯 Objectives

To analyze customer data and identify meaningful customer segments
To implement explainable rule-based segmentation
To apply unsupervised machine learning (K-Means) for data-driven clustering
To visualize customer clusters using PCA
To build an interactive UI for analysis and result exploration
To maintain a modular architecture for future upgrades (DB, APIs, Cloud)

🧠 Segmentation Approaches Used 1️⃣ Rule-Based Segmentation

Uses predefined business rules
Based on Annual Income and Spending Score
Produces interpretable segments:
High Value Customers
Medium Value Customers
Low Value Customers

2️⃣ Machine Learning–Based Segmentation

Uses K-Means clustering
Unsupervised learning (no labels required)
Discovers hidden patterns in customer behavior
Cluster count (K) selectable via UI

📊 Visualization Techniques

Bar Chart: Shows distribution of customers across segments

PCA Scatter Plot:

Reduces high-dimensional data to 2D
Used only for visualization
Helps understand cluster separation visually
Clustering is performed on original scaled features; PCA is not used for training.

🏗️ System Architecture

The project follows a layered architecture:
UI Layer Streamlit-based interface for file upload, configuration, and visualization
Controller Layer Orchestrates preprocessing, segmentation, and result aggregation
Service Layer
Preprocessing
Feature engineering
ML training & prediction
PCA transformation
Storage Layer
CSV files for uploaded data
Pickle files for trained ML models
This separation ensures low coupling and high maintainability.

📁 Project Structure customer_segmentation/ │ ├── ui/ │ └── app.py # Streamlit UI (presentation layer only) │ ├── backend/ │ ├── services/ │ │ ├── preprocessing.py │ │ ├── feature_engineering.py │ │ ├── ml_train.py │ │ ├── ml_predict.py │ │ ├── pca_visualization.py │ │ │ ├── storage/ │ │ └── file_store.py │ │ │ └── controller.py # Orchestrates system flow │ ├── models/ │ └── kmeans.pkl # Trained ML model │ ├── data/ │ └── uploaded.csv # Uploaded dataset │ ├── requirements.txt └── README.md

📂 Dataset Used

Mall Customers Dataset

Features:

Age
Annual Income (k$)
Spending Score (1–100)
The dataset represents customer demographics and purchasing behavior and is commonly used for customer analytics.

⚙️ Preprocessing Steps

Feature selection
Handling missing values
Feature scaling using StandardScaler
Preparation of data for fair distance-based clustering

🖥️ User Interface Features

Upload customer dataset (CSV)
Select segmentation type:
Rule-Based
ML-Based (K-Means)
Select number of clusters (K)

View:

Segmented customer table
Segment distribution chart
PCA cluster visualization
Download segmented dataset
🚀 How to Run the Project 1️⃣ Install Dependencies pip install -r requirements.txt

2️⃣ Run the Application

From the project root directory:
python -m streamlit run ui/app.py

🎓 Academic Highlights

Combines business logic and machine learning
Uses unsupervised learning
Maintains explainability
Implements clean software architecture
Visualization-driven interpretation

🔮 Future Scope (Semester 8)

Database integration (PostgreSQL / MySQL)
REST API using FastAPI
Cloud deployment
Real-time customer segmentation
Recommendation systems
Advanced clustering algorithms

🏁 Conclusion

This project demonstrates how machine learning and feature engineering can be effectively combined to perform customer segmentation. The modular design allows easy extension while maintaining clarity and interpretability, making it suitable for both academic evaluation and real-world applications.
