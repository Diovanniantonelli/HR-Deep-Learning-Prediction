# 💼 HR Deep Learning Prediction

🚀 Projeto de Machine Learning para auxiliar o **departamento de Recursos Humanos** na **previsão de rotatividade de funcionários**, ajudando a reduzir **custos de contratação** e **treinamento**.

---

## 📊 Objetivo

Desenvolver modelos de Inteligência Artificial capazes de prever a probabilidade de um colaborador deixar a empresa, utilizando dados reais de Recursos Humanos. Essa solução pode ser aplicada para:

- Reduzir o turnover;
- Melhorar a retenção de talentos;
- Otimizar estratégias de RH;
- Apoiar decisões gerenciais com dados.

---

## 🧠 Técnicas Utilizadas

- Análise Exploratória de Dados (EDA) com `Seaborn` e `Matplotlib`;
- Pré-processamento: `OneHotEncoder`, `MinMaxScaler`;
- Modelos treinados:
  - **Regressão Logística**
  - **Random Forest**
  - **Rede Neural (Keras + TensorFlow)**
- Validação com:
  - Acurácia
  - Matriz de Confusão
  - F1-Score, Precisão, Recall

---

## 🛠️ Tecnologias

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras
- Pickle (serialização de modelo)

---

## 🗂️ Estrutura do Projeto

HR-Deep-Learning-Prediction/
│
├── Human_Resources.csv # Base de dados utilizada
├── HR_prediction.ipynb # Jupyter Notebook com o projeto completo
├── variaveis_modelo.pkl # Modelo salvo (scaler, encoder e regressão)
└── README.md # Descrição do projeto


---

## 📈 Resultados

- Acurácia do modelo com Regressão Logística: **85%**
- F1-score com Rede Neural: **~63% (macro)**
- Identificação clara de variáveis com maior impacto na saída do funcionário (ex: OverTime, DistanceFromHome, JobRole)

---

## 🤖 Modelo Final

A rede neural possui 3 camadas ocultas com 25 neurônios cada e função de ativação ReLU, finalizando com uma camada sigmoide para saída binária (previsão de saída ou permanência).

```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(25, activation='relu', input_shape=(50,)))
model.add(tf.keras.layers.Dense(25, activation='relu'))
model.add(tf.keras.layers.Dense(25, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
