
### 🧠 ML DevOps Engineer Full-Time Learning Plan (6 Months)

**Start Date:** May 26, 2025
**Goal:** Become job-ready for MLOps / ML DevOps roles
**Schedule:** 8 hrs/day, Mon–Fri (\~1040+ hours over 6 months)
---

## 📆 Week 1: Python & Data Foundations 👩🏼‍💻

**Goal:** Get comfortable with Python for data science and start exploring Jupyter, NumPy, Pandas.

| Day | Topics           | Links / Tasks                                                                                                                                             |
| --- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mon | Python refresher | - [Google's Python class](https://developers.google.com/edu/python) <br> - Practice in [Jupyter](https://jupyter.org/try) or VSCode            |
| Tue | NumPy            | - [NumPy Tutorial](https://numpy.org/learn/) <br> - Exercises from [W3 NumPy](https://www.w3schools.com/python/numpy/)                                    |
| Wed | Pandas           | - [Pandas 10-Min Guide](https://pandas.pydata.org/docs/user_guide/10min.html) <br> - Work with [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic) |
| Thu | Data analysis    | - Create simple data analysis notebook <br> - Use [Seaborn](https://seaborn.pydata.org/) for visualizations                                               |
| Fri | Project Day      | - 🛠️ Build: Titanic EDA notebook (load, clean, visualize, summarize) <br> - Push to GitHub with README                                                   |

---

## 📆 Week 2: Machine Learning Core Concepts

**Goal:** Learn ML theory and implement basic models using `scikit-learn`.

| Day | Topics                     | Links / Tasks                                                                                                                                 |
| --- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Mon | Supervised vs Unsupervised | - [Google Crash Course - Intro to ML](https://developers.google.com/machine-learning/crash-course/ml-intro)                                   |
| Tue | Classification             | - [scikit-learn Classification Guide](https://scikit-learn.org/stable/supervised_learning.html)                                               |
| Wed | Regression, Metrics        | - Implement Linear & Logistic Regression <br> - Learn about accuracy, precision, recall, F1                                                   |
| Thu | Model validation           | - Train/test split, cross-validation <br> - Use [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) |
| Fri | Project Day                | - 🛠️ Build: Classification model on Iris/Titanic <br> - Include confusion matrix & evaluation metrics                                        |

---

## 📆 Week 3: Intro to MLOps Concepts + GitHub Setup

**Goal:** Learn what MLOps is and start using real project structure and automation.

| Day | Topics                 | Links / Tasks                                                                                                                        |
| --- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Mon | What is MLOps?         | - [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) |
| Tue | Project structure      | - Learn cookiecutter data science pattern ([Repo](https://github.com/drivendata/cookiecutter-data-science))                          |
| Wed | Git, GitHub, Makefiles | - Add Makefile and DVC (next week) <br> - Create `mlops-portfolio` GitHub repo                                                       |
| Thu | CI/CD in ML            | - Read [CI/CD for ML](https://ml-ops.org/content/cicd) <br> - Intro to GitHub Actions                                                |
| Fri | Project Day            | - 🛠️ Refactor previous week’s model into a repo with `src/`, `notebooks/`, `Makefile`                                               |

---

## 📆 Week 4: DVC, MLflow, Experiment Tracking

**Goal:** Manage datasets & experiments like a real ML engineer.

| Day | Topics                  | Links / Tasks                                                                                           |
| --- | ----------------------- | ------------------------------------------------------------------------------------------------------- |
| Mon | DVC for data versioning | - [DVC Intro](https://dvc.org/doc/start) <br> - Apply to Titanic/Iris project                           |
| Tue | MLflow tracking         | - [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html) <br> - Log parameters and metrics |
| Wed | Integrate MLflow + DVC  | - Connect DVC + MLflow <br> - Use GitHub Actions to automate                                            |
| Thu | Model packaging         | - Learn [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)                     |
| Fri | Project Day             | - 🛠️ Upgrade last week’s project with DVC, MLflow logging & GitHub Actions CI                          |

---

## 📅 Week 5: Docker for ML

**Goal:** Containerize ML workflows with Docker.

| Day | Task                                                              | Resources                                                                                              |
| --- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Mon | Intro to Docker                                                   | [Docker Crash Course](https://docker-curriculum.com/)                                                  |
| Tue | Dockerfiles & Images                                              | [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) |
| Wed | Dockerizing Python Apps                                           | [Docker Python Example](https://docs.docker.com/samples/python/)                                       |
| Thu | Docker Compose                                                    | [Docker Compose Docs](https://docs.docker.com/compose/)                                                |
| Fri | 🛠️ Project: Dockerize previous ML model and run training locally |                                                                                                        |

## 📅 Week 6: Kubernetes Basics

**Goal:** Learn how to deploy containers using Kubernetes.

| Day | Task                                             | Resources                                                                                       |
| --- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| Mon | K8s Intro + Architecture                         | [Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/)                    |
| Tue | Pods, Deployments                                | [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) |
| Wed | Services, Ingress                                | [K8s Networking Guide](https://kubernetes.io/docs/concepts/services-networking/)                |
| Thu | Minikube + kubectl                               | [Minikube Docs](https://minikube.sigs.k8s.io/docs/start/)                                       |
| Fri | 🛠️ Project: Deploy Dockerized model to Minikube |                                                                                                 |

## 📅 Week 7: Airflow for ML Pipelines

**Goal:** Build training pipelines with Airflow.

| Day | Task                                                       | Resources                                                                                             |
| --- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Mon | Intro to Airflow                                           | [Airflow Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html)               |
| Tue | DAGs and Operators                                         | [Airflow DAG Concepts](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html) |
| Wed | Triggering and Scheduling                                  | [Airflow Scheduling](https://airflow.apache.org/docs/apache-airflow/stable/scheduler.html)            |
| Thu | ML pipeline in Airflow                                     | [ML with Airflow Example](https://github.com/apache/airflow/tree/main/airflow/example_dags)           |
| Fri | 🛠️ Project: Create full training-eval pipeline in Airflow |                                                                                                       |

## 📅 Week 8: Model Serving + REST APIs

**Goal:** Learn serving models via web APIs.

| Day | Task                                                      | Resources                                                                                           |
| --- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Mon | Flask + FastAPI intro                                     | [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)                                          |
| Tue | Model loading and prediction endpoint                     | [Serving ML with FastAPI](https://towardsdatascience.com/fastapi-for-machine-learning-5ec4c0232dd6) |
| Wed | Testing and Swagger                                       | [FastAPI Docs](https://fastapi.tiangolo.com/)                                                       |
| Thu | Dockerize your API                                        | Reuse Docker knowledge from Week 5                                                                  |
| Fri | 🛠️ Project: Build + serve ML model via FastAPI container |                                                                                                     |

## 📅 Week 9: Cloud Platforms (AWS or GCP)

**Goal:** Deploy ML APIs to the cloud.

| Day | Task                                          | Resources                                                                                             |
| --- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Mon | IAM, EC2 basics (AWS) or Compute Engine (GCP) | [AWS EC2 Tutorial](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)           |
| Tue | Storage: S3 or GCS                            | [AWS S3](https://docs.aws.amazon.com/s3/index.html), [GCP GCS](https://cloud.google.com/storage/docs) |
| Wed | Deploy Docker container to cloud              | [Deploy Docker to EC2](https://docs.docker.com/cloud/ecs-integration/)                                |
| Thu | Test/monitor live API                         | Use Postman, CloudWatch/GCP Logs                                                                      |
| Fri | 🛠️ Project: Deploy model API to AWS/GCP      |                                                                                                       |

## 📅 Week 10: CI/CD for ML

**Goal:** Automate model training and deployment.

| Day | Task                                                       | Resources                                                 |
| --- | ---------------------------------------------------------- | --------------------------------------------------------- |
| Mon | CI/CD principles                                           | [CI/CD for ML](https://ml-ops.org/content/cicd)           |
| Tue | GitHub Actions deep dive                                   | [GitHub Actions Docs](https://docs.github.com/en/actions) |
| Wed | Training in CI                                             | Use Makefile + script in GitHub workflow                  |
| Thu | Auto-deploy API in CI                                      | Add deployment step to GitHub Actions                     |
| Fri | 🛠️ Project: End-to-end CI/CD pipeline with GitHub Actions |                                                           |

## 📅 Week 11: Model Monitoring

**Goal:** Monitor drift, latency, performance in prod.

| Day | Task                                                  | Resources                                                                 |
| --- | ----------------------------------------------------- | ------------------------------------------------------------------------- |
| Mon | Monitoring basics                                     | [Prometheus + Grafana](https://prometheus.io/docs/visualization/grafana/) |
| Tue | Model drift detection                                 | [Alibi Detect](https://docs.seldon.io/projects/alibi-detect/en/stable/)   |
| Wed | Log structured metrics                                | Use `logging` + Prometheus exporters                                      |
| Thu | Visualize metrics                                     | Build Grafana dashboard locally                                           |
| Fri | 🛠️ Project: Add monitoring to API deployed in Week 9 |                                                                           |

---

## 📅 Week 12: Weights & Biases (W\&B)

**Goal:** Integrate W\&B for experiment tracking, logging, and collaboration.

| Day | Task                                                      | Resources                                                                |
| --- | --------------------------------------------------------- | ------------------------------------------------------------------------ |
| Mon | Intro to W\&B, install/setup                              | [W\&B Quickstart](https://docs.wandb.ai/quickstart)                      |
| Tue | Logging metrics during training                           | [W\&B Keras/PyTorch examples](https://docs.wandb.ai/guides/integrations) |
| Wed | Model comparison dashboards                               | [W\&B Reports](https://docs.wandb.ai/guides/reports)                     |
| Thu | Team collaboration & Sweeps                               | [W\&B Sweeps](https://docs.wandb.ai/guides/sweeps)                       |
| Fri | 🛠️ Project: Add W\&B tracking to past training pipelines |                                                                          |

## 📅 Week 13: SageMaker 101 (AWS ML Platform)

**Goal:** Train, deploy, and monitor ML models using SageMaker.

| Day | Task                                            | Resources                                                                                             |
| --- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Mon | Intro + SageMaker Studio setup                  | [SageMaker Studio Setup](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio.html)              |
| Tue | Build & train models                            | [SageMaker Training Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html) |
| Wed | Deploy + invoke endpoints                       | [SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)        |
| Thu | Monitor deployed models                         | [Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)                   |
| Fri | 🛠️ Project: Train + deploy model via SageMaker |                                                                                                       |

## 📅 Week 14: TensorRT and Model Optimization

**Goal:** Accelerate inference for deployment.

| Day | Task                                                 | Resources                                                                                           |
| --- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Mon | Intro to model compression + TensorRT                | [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)                                            |
| Tue | Convert models to ONNX + optimize                    | [ONNX Export](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)        |
| Wed | Benchmark speedup results                            | Use Jupyter or script benchmarks                                                                    |
| Thu | TensorRT with Docker                                 | [TensorRT in Docker](https://docs.nvidia.com/deeplearning/tensorrt/docker/docker-installation.html) |
| Fri | 🛠️ Project: Optimize + deploy a model with TensorRT |                                                                                                     |

## 📅 Week 15: DataOps for ML

**Goal:** Automate versioning, validation, ingestion pipelines.

| Day | Task                                                    | Resources                                                                           |
| --- | ------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Mon | Data validation + schema checks                         | [Great Expectations](https://greatexpectations.io/)                                 |
| Tue | Data versioning                                         | [DVC Pipelines](https://dvc.org/doc/start/data-versioning)                          |
| Wed | Automate ingestion jobs                                 | [Airflow + DataOps](https://www.astronomer.io/guides/data-ingestion-with-airflow/)  |
| Thu | Batch vs stream ingestion                               | [Batch vs Stream](https://cloud.google.com/architecture/batch-vs-stream-processing) |
| Fri | 🛠️ Project: End-to-end data validation + ingestion DAG |                                                                                     |

## 📅 Week 16: Advanced MLOps Architecture

**Goal:** Understand and design complete MLOps pipelines.

| Day | Task                                                    | Resources                                                                                 |
| --- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Mon | Study major MLOps patterns                              | [MLOps Stack Guide](https://ml-ops.org/content/mlops-stack)                               |
| Tue | Kubeflow intro                                          | [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/v1/introduction/) |
| Wed | TFX + Vertex AI overview                                | [TFX Overview](https://www.tensorflow.org/tfx)                                            |
| Thu | Pick stack for capstone project                         |                                                                                           |
| Fri | 🛠️ Project: Architecture doc + tech stack for capstone |                                                                                           |

## 📅 Week 17–21: Capstone Part 1 — Design + Setup

**Goal:** Prepare your full-stack ML app project for capstone.

| Week | Focus                        | Tasks                                                    |
| ---- | ---------------------------- | -------------------------------------------------------- |
| 17   | Data Collection & Cleaning   | Use real dataset (e.g., Kaggle), apply DVC, validation   |
| 18   | Model Dev + W\&B integration | Train model (classification or regression), log via W\&B |
| 19   | Serve Model API              | FastAPI + Docker + CI/CD                                 |
| 20   | Deploy to Cloud              | Use AWS/GCP + Monitoring tools                           |
| 21   | Evaluation + Monitoring      | Implement Grafana or SageMaker Monitor + test logs       |

## 📅 Week 22: Capstone Part 2 — Automation

**Goal:** Add retraining pipelines, batch/online deployment.

| Day | Task                                                            | Resources                                                                                 |
| --- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Mon | Automate pipeline in Airflow/Kubeflow                           | [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/v1/introduction/) |
| Tue | CI/CD for retraining                                            | Use GitHub Actions for pipeline triggers                                                  |
| Wed | Auto-monitor drift + retrigger training                         | Integrate Alibi Detect                                                                    |
| Thu | Publish architecture + README                                   | Document clearly on GitHub                                                                |
| Fri | 🛠️ Final Test: simulate end-to-end model retraining + redeploy |                                                                                           |

## 📅 Week 23: Edge Deployment with TensorFlow Lite + CoreML

**Goal:** Learn to deploy models on edge devices (mobile, embedded).

| Day | Task                                                | Resources                                                                                                    |
| --- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Mon | Intro to TFLite, convert model                      | [TFLite Conversion](https://www.tensorflow.org/lite/convert)                                                 |
| Tue | Optimize + quantize model                           | [Post-training quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) |
| Wed | Integrate TFLite with Android/iOS                   | [TFLite Mobile Deployment](https://www.tensorflow.org/lite/guide/android)                                    |
| Thu | CoreML intro + converter                            | [CoreML Tools](https://github.com/apple/coremltools)                                                         |
| Fri | 🛠️ Project: Edge model prototype for mobile device |                                                                                                              |

## 📅 Week 24: Federated Learning + Privacy

**Goal:** Explore decentralized model training and privacy preservation.

| Day | Task                                                     | Resources                                                           |
| --- | -------------------------------------------------------- | ------------------------------------------------------------------- |
| Mon | Intro to FL + use cases                                  | [TensorFlow Federated](https://www.tensorflow.org/federated)        |
| Tue | Simulate FL across clients                               | [TFF Tutorials](https://www.tensorflow.org/federated/tutorials)     |
| Wed | Explore privacy-preserving training                      | [Opacus (PyTorch)](https://opacus.ai/)                              |
| Thu | Add differential privacy to pipeline                     | [Google DP Library](https://github.com/google/differential-privacy) |
| Fri | 🛠️ Project: FL training simulation + privacy evaluation |                                                                     |

## 📅 Week 25: Multi-Model Pipelines + Ensemble Strategies

**Goal:** Implement and deploy advanced inference setups.

| Day | Task                                                          | Resources                                                                 |
| --- | ------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Mon | Learn ensemble theory + voting                                | [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) |
| Tue | Pipeline with multiple models + ensembling                    | Use scikit-learn + FastAPI                                                |
| Wed | Ensemble with different architectures                         | Train different models, merge predictions                                 |
| Thu | Optimize ensemble latency                                     | Batch predictions + async API testing                                     |
| Fri | 🛠️ Project: Deploy model ensemble for fraud detection or NLP |                                                                           |

## 📅 Week 26: ML Monitoring & Feedback Loops

**Goal:** Build a production-ready monitoring dashboard + feedback system.

| Day | Task                                                         | Resources                                                             |
| --- | ------------------------------------------------------------ | --------------------------------------------------------------------- |
| Mon | Integrate Prometheus + Grafana for monitoring                | [Prometheus Setup](https://prometheus.io/docs/introduction/overview/) |
| Tue | Log model predictions + performance                          | Use custom logger or Grafana Loki                                     |
| Wed | Detect model drift in real-time                              | [Evidently AI](https://www.evidentlyai.com/)                          |
| Thu | Design feedback loop architecture                            | Define retraining/alert criteria                                      |
| Fri | 🛠️ Final Project: Full feedback-aware ML system + dashboard |                                                                       |

---
