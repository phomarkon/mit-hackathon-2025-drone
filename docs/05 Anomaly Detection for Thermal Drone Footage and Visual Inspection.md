**Anomaly Detection for Thermal Drone Footage and Visual Inspection**  
 **Track:** Fine-Tuning of Models

---

## **1\. Goals / Motivation**

Manual review of visual data—whether from drone imagery in search-and-rescue missions or product inspection on factory floors—is slow, error-prone, and expensive. Meanwhile, **state-of-the-art anomaly detection models** have achieved outstanding performance in industrial quality assurance but are rarely applied to humanitarian or non-industrial use cases.

This challenge invites participants to **repurpose these models for new domains**—with a special focus on building a prototype that helps identify **missing persons in thermal drone footage** or **detect defects in industrial inspection tasks**.

Participants can choose either domain or explore both. The broader goal is to show that **AI techniques originally developed for one sector (industry)** can be **transferred and fine-tuned** for meaningful, high-impact use elsewhere—especially in **humanitarian emergencies**.

---

## **2\. Features**

Participants are expected to develop a working prototype that applies **unsupervised anomaly detection** to one of two real-world domains:

### **A. Search & Rescue – Thermal Drone Imagery**

Design a prototype that:

* **Classifies thermal drone images** as “normal” (no person) or “abnormal” (potential human presence).

* **Prioritizes abnormal findings** so that rescue workers can focus on the most relevant frames first.

* **Supports image upload or simulated real-time streams**, allowing teams to process entire drone footage folders.

**Optional Extensions:**

* Integrate location metadata or GPS overlays.

* Simulate alerts for emergency response teams (e.g., via dashboard, email, or SMS).

### **B. Industrial Visual Inspection**

Alternatively, develop a tool for detecting visual defects in industrial settings:

* **Classify input images** (e.g., of circuit boards, automotive parts) into “normal” or “defective.”

* **Provide batch upload and reporting** tools that summarize anomalies.

* **Highlight defects visually** (e.g., via heatmaps or bounding boxes) to aid quality inspectors.

**Optional Extensions:**

* Include model performance metrics in the UI.

* Connect to a simulated factory line camera feed.

### **Common Functional Requirements (both domains):**

* A **basic user interface** (web or command-line) for uploading and processing images.

* A **prioritization mechanism** (e.g., confidence scores, sorted outputs).

* Use of **transfer learning**: adapt existing models trained on datasets like MVTec AD to new target domains.

* **Visual feedback** (optional but encouraged): bounding boxes, saliency maps, or pixel anomaly heatmaps.

---

## **3\. Hints and Resources**

### **Datasets**

Link to a test Dataset: [AI Dataset Drone Image detection](https://drive.google.com/drive/mobile/folders/1V6qOwVNxbWbTJk8s5H6BNRHh2U1hsxuQ/1oFkzxMvlCA5Az4arGLqv3qMtJ7mlimeV/1563VLi02jfGnfIBPUpM6PJAboECoCank/11To04t0ytnoBmL-mwEivrb8pzL5OUBGY?usp=share_link&sort=13&direction=a)

Link to full dataset: [AI Dataset Drone Image detection](https://drive.google.com/drive/mobile/folders/1V6qOwVNxbWbTJk8s5H6BNRHh2U1hsxuQ?usp=share_link)

This dataset consists of 226 thermal images using a DJI Matrice 300 (M300) drone equipped with the Zenmuse H20T thermal imaging camera. Images were taken with a camera angle of 90 degrees and a flight altitude of approximately 80 meters above ground level (AGL). The ambient temperature during data collection was 11°C.

The dataset is organized as follows:

166 Normal Training Images  
These images represent scenarios without any unusual thermal signatures.

60 Test Images  
Equally split into:  
30 Normal test images  
30 Abnormal test images (Including thermal signatures.) 

The dataset is provided by the Black Forest Mountain Rescue Team and are provided for the Global MIT Hackathon Challenge for the purpose to train a model to detect missing people during rescue mission’s in a mountain area.

* **Industrial domain**:

  * [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) (standard benchmark for anomaly detection).

* **Thermal rescue domain**:

  * No public dataset currently exists. Simulate using public thermal images, open drone datasets, or generate mock thermal images with synthetic anomalies.

### **Models**

Start with pre-trained models from:

* PapersWithCode – Anomaly Detection

* Notable models: **PatchCore**, **FastFlow**, **fAnoGAN**, **DRAEM**, **PaDiM**

* Frameworks: PyTorch, TensorFlow

### **Transfer Learning**

* Research from MIT and others confirms strong results from adapting industrial-trained anomaly models to novel domains.

* Fine-tune or adapt pre-trained models using even small new datasets.

### **Technical Tools**

* **Image processing**: OpenCV

* **Web scraping (if collecting new images)**: Selenium, BeautifulSoup

* **Prototyping**: Streamlit, Gradio, Flask, or FastAPI

* **Visualization**: Matplotlib, Seaborn, Plotly for heatmaps and dashboards

---

## **4\. Evaluation Criteria**

Judges will evaluate submissions on:

* **Anomaly Detection Effectiveness**  
   How well does the system distinguish between normal and abnormal images? Are false positives/negatives minimized?

* **Domain Adaptation & Generalization**  
   How well was a model originally trained on another domain adapted to this new application?

* **User-Centered Design & Usability**  
   Is the interface intuitive and helpful for non-technical users (e.g., rescue teams or factory inspectors)?

* **Creativity & Innovation**  
   Does the solution include novel approaches, such as automation, alert systems, or integrations?

* **Clarity & Documentation**  
   Is the code well-documented and the overall approach clearly explained?

---

## **5\. Why Does It Matter?**

### **Humanitarian Impact**

In a natural disaster, every minute counts. AI-assisted drone analysis can cut down the time needed to locate missing persons, making rescue operations faster, safer, and more efficient. An open-source tool like this could become a powerful asset for emergency response teams worldwide.

### **Industrial Efficiency**

Defect detection is a multi-billion-dollar challenge in manufacturing. Automating it not only reduces labor and cost but also improves safety and product quality. This challenge allows participants to explore high-value use cases for anomaly detection in a practical, scalable format.

Ultimately, this project highlights how **AI can amplify human judgment**, not replace it—whether in saving lives or ensuring product excellence.

