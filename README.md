Predição de AVC com Machine Learning

Projeto desenvolvido na pós-graduação com foco em análise de dados e modelos de classificação para prever risco de AVC.

Tecnologias
- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- Streamlit

Etapas do Projeto
- Limpeza de dados
- Feature engineering (BMI, idade, pressão)
- Treinamento do modelo
- Avaliação (Acurácia, ROC, AUC)
- Interpretabilidade com SHAP

Resultados
- Acurácia: 72%
- AUC: 0.80
- F1-score: 0.73
- Recall: 0.80
O modelo apresentou bom desempenho na identificação de casos positivos, com destaque para o alto recall, importante em cenários de saúde.

Como rodar

```bash
pip install -r requirements.txt
streamlit run app/app.py
