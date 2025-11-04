````markdown
<h2 align="center"> Demostración del Proyecto</h2>

<p align="center">
  <img src="assets/demostracion.gif" width="600" alt="Vista previa de la API">
</p>

# Proyecto Final MLOps Local — Predicción de Enfermedades Cardíacas

Este proyecto fue desarrollado como parte del curso de **Machine Learning / MLOps** de la **Universidad del Norte**.  
Nuestro objetivo fue construir un flujo **MLOps completo en entorno local**, utilizando herramientas open source como **FastAPI, Docker, Kubernetes (Minikube)** y **Evidently**, para entrenar, desplegar y monitorear un modelo de aprendizaje automático capaz de **predecir la probabilidad de enfermedad cardíaca** a partir de variables clínicas.

---

## **Descripción general del proyecto**

El flujo completo abarca las siguientes etapas:

1. **Entrenamiento del modelo**
   - Limpieza, codificación y análisis del dataset original de enfermedades cardíacas.
   - Entrenamiento con varios clasificadores: *Logistic Regression (con L1/L2)*, *Random Forest*, *K-Neighbors*, *XGBoost*, *Naive Bayes* y *Ridge Classifier*.
   - Selección del mejor modelo en base al puntaje AUC.
   - Exportación del modelo entrenado (`model.joblib`) y del esquema de variables (`training_columns.json`).

2. **Despliegue con FastAPI**
   - Se construyó una API REST con endpoints `/` (raíz), `/health` (verificación del estado del servicio) y `/predict` (predicción de enfermedad cardíaca).
   - El endpoint `/predict` recibe datos clínicos en formato JSON y devuelve la probabilidad de enfermedad cardíaca.

3. **Contenerización con Docker**
   - Se creó un `Dockerfile` para construir una imagen reproducible que permite ejecutar el modelo sin dependencias externas.
   - Esto garantiza portabilidad entre entornos.

4. **Orquestación con Kubernetes (Minikube)**
   - Los archivos `deployment.yaml` y `service.yaml` permiten desplegar el contenedor en un clúster local administrado con Minikube.
   - Se exponen los puertos y servicios para acceder a la API desde el navegador.

5. **Integración Continua (CI)**
   - Se configuró GitHub Actions para ejecutar automáticamente el *linting* con `flake8` y las pruebas unitarias con `pytest` cada vez que se realiza un `push`.

6. **Monitoreo de deriva de datos**
   - Se implementó un reporte con **Evidently** para analizar el posible cambio en las distribuciones de los datos entre el entrenamiento y los datos actuales.
   - Este reporte se guarda como `drift_report.html`.

---

## **Tecnologías empleadas**

| Componente | Herramienta / Librería |
|-------------|------------------------|
| Lenguaje principal | Python 3.10 |
| Framework API | FastAPI |
| Modelado ML | Scikit-learn, XGBoost |
| Contenedores | Docker |
| Orquestación | Kubernetes (Minikube) |
| Pruebas | Pytest |
| Estilo de código | Flake8 |
| Monitoreo | Evidently |
| Integración continua | GitHub Actions |


## **Ejecución local (FastAPI sin Docker)**

Para probar la API directamente desde el entorno local:

1. Activar el entorno virtual:
   ```bash
   conda activate ml_venv
````

2.  Ejecutar el servidor FastAPI:

    ```bash
    uvicorn app.api:app --reload
    ```

3.  Abrir el navegador en la URL:
    [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

> **Nota para la revisión:**
> Para facilitar la prueba de la API en la interfaz `/docs`, hemos creado profe el archivo `ejemplos_de_prueba.docx`. Este documento contiene la explicación detallada de cada parámetro que recibe el modelo y varios casos de prueba listos para probar, los cuales usted puede modificar tambien.

-----

## **Ejecución con Docker**

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

Luego acceder a:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

-----

## **Despliegue local con Kubernetes (Minikube)**

```bash
minikube start
minikube image load heart-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
minikube service heart-service --url
```

Ejemplo de salida:

```
[http://127.0.0.1:64642](http://127.0.0.1:64642)
```

-----

## **Pruebas unitarias (Pytest)**

Ejecución de pruebas automáticas:

```bash
pytest -v
```

Salida esperada:

```
tests/test_api.py::test_root PASSED
tests/test_api.py::test_health PASSED
tests/test_api.py::test_predict_valid_input PASSED
```

-----

## **Monitoreo de deriva de datos (Evidently)**

Ejecutar en Python:

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
import joblib
import json

with open("training_columns.json") as f:
   cols = json.load(f)

model = joblib.load("model.joblib")
df = pd.read_csv("heart.csv")

reference = df.sample(200, random_state=42)
current = df.sample(200, random_state=24)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)
report.save_html("drift_report.html")
```

-----

## **Integración Continua (GitHub Actions)**

```yaml
name: CI - Heart Disease API

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r docker/requirements.txt
          pip install flake8 pytest httpx

      - name: Lint code
        run: flake8 app/

      - name: Run tests
        run: pytest -q
```

-----

## **Estructura del proyecto**

```
heart-disease-mlops/
│
├── app/
│   └── api.py
│
├── docker/
│   ├── Dockerfile
│   ├── requirements.txt
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│
├── notebooks/
│   └── (notebooks de entrenamiento y EDA)
│
├── tests/
│   └── test_api.py
│
├── model.joblib
├── training_columns.json
├── drift_report.html
├── .flake8
├── .dockerignore
├── README.md
└── .github/workflows/ci.yml
```

-----

## **Autores del proyecto**

  * **Johan David Díaz López**
  * **Luis Peñaranda**
  * **Miguel Lugo**
  * **Héctor San Juan**

Universidad del Norte — Programa de Ciencia de Datos
Miniproyecto: *Machine Learning / MLOps Local*

-----

## **Conclusión general**

Durante el desarrollo de este proyecto aprendimos a implementar las etapas esenciales del ciclo de vida de un modelo de Machine Learning bajo prácticas de **MLOps local**, desde el preprocesamiento de datos y el entrenamiento, hasta el despliegue y monitoreo.

Además, logramos integrar herramientas de orquestación (**Kubernetes**), contenerización (**Docker**) y pruebas automatizadas (**CI/CD**) para asegurar **reproducibilidad, calidad y estabilidad del modelo**.

-----
