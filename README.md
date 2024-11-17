# Implementação e Análise do Algoritmo de Regressão Linear

## Descrição do Projeto
Este projeto tem como objetivo implementar um modelo de **Regressão Linear** para prever a **taxa de engajamento** dos principais influenciadores do Instagram. Utilizando métricas como número de seguidores, média de curtidas e engajamento em novas postagens, buscamos entender os fatores que mais influenciam o engajamento.

O projeto inclui desde a análise exploratória dos dados até a construção, validação e avaliação do modelo, com visualizações e métricas que demonstram seu desempenho.

---

## Instalação
Siga os passos abaixo para configurar o ambiente e rodar o projeto:

1. **Clone o repositório**:
   ```bash
   git clone <URL do repositório>
   cd <nome_da_pasta>

2. **Crie e ative um ambiente virtual:** 

    ```bash
    python -m venv env
    source env/bin/activate  # No Windows: env\Scripts\activate

3. **Instale as dependências:**

    ```bash
    pip install -r requirements.txt

## Como Executar

1. **Certifique-se de que o arquivo de dados *(top_insta_influencers_data.csv)* está no diretório raiz do projeto.**

2. **Execute o script principal:**
    ```bash
    python src/main.py

3. **O programa realizará as seguintes etapas:**
- Carregamento e limpeza do conjunto de dados.
- Análise exploratória e visualização dos dados.
- Treinamento e avaliação do modelo de Regressão Linear.
- Geração de gráficos e exibição das métricas de desempenho.

## Estrutura dos Arquivos

    /docs
        ├── Relatorio_Tecnico_RegLinear.pdf    # Relatório técnico completo
    /src
        ├── main.py                            # Código principal do projeto
        ├── utils.py                           # Funções auxiliares para limpeza e análise
        ├── requirements.txt                   # Lista de dependências do projeto
    /data
        ├── top_insta_influencers_data.csv     # Conjunto de dados usado no projeto
    README.md                                 # Este arquivo

## Tecnologias Utilizadas

**Linguagem:** *Python*

**Bibliotecas:**

- *pandas:* Manipulação e análise de dados.
- *numpy:* Operações matemáticas e numéricas.
- *matplotlib* e *seaborn:* Visualização de dados.
- *scikit-learn:* Implementação do modelo de Regressão Linear.

## Autores:

- **Marley Rebouças Ferreira**
- **Murilo Carlos Novais**



