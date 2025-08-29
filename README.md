Sistema de Trading Algorítmico com IA para Criptomoedas
Este repositório contém o código-fonte de um ecossistema completo para trading algorítmico no mercado de criptomoedas, utilizando Machine Learning para prever movimentos de preço e executar operações de forma autônoma na exchange Bybit.

? Visão Geral do Projeto
O sistema é composto por múltiplos módulos que trabalham em conjunto para cobrir todo o ciclo de vida de uma estratégia de trading quantitativo:

Engenharia de Features: Criação de um conjunto rico de indicadores técnicos a partir de dados históricos.

Treinamento de Modelo: Treinamento de modelos de Machine Learning (XGBoost) especializados para cada par de criptomoedas.

Backtesting e Otimização: Validação rigorosa da estratégia em dados passados e otimização de hiperparâmetros para maximizar o retorno ajustado ao risco.

Execução em Tempo Real: Um robô que consome as predições do modelo, gerencia risco e executa ordens na Bybit.

Monitoramento e Controle: Um painel web interativo para acompanhar a performance, visualizar posições e intervir manualmente, se necessário.

?? Arquitetura
O projeto é modularizado para garantir clareza, manutenção e escalabilidade:

feature_engineering.py: Módulo central para calcular todos os indicadores técnicos (EMAs, RSI, ADX, Bandas de Bollinger, etc.), garantindo consistência entre o treinamento e a execução.

model_trainer.py: Script responsável por treinar um modelo XGBoost para cada ativo, criando um "artefato" (.pkl) que empacota o modelo, a lista de features e a função de pré-processamento.

inference_server.py: Um servidor de API (Flask) que carrega os modelos treinados e expõe um endpoint /predict para fornecer previsões em tempo real, desacoplando a IA do robô principal.

backtest.py: Ferramenta para simular a execução da estratégia em dados históricos, considerando custos realistas como taxas e slippage.

otimizador.py: Utiliza Grid Search para testar múltiplas combinações de parâmetros (ex: limiar de confiança do modelo, limiar de ADX) e encontrar a configuração mais lucrativa.

bot_bybit.py: O coração do sistema. Este robô orquestra todo o processo em tempo real: busca dados, consulta a API de inferência, calcula o tamanho da posição com base no risco, executa ordens e gerencia o Trailing Stop Loss em uma thread separada.

painel.py: Uma interface de usuário (NiceGUI) que fornece um dashboard completo para monitoramento, com gráficos de performance, visualização de posições abertas e logs em tempo real.

?? Tecnologias Utilizadas
Linguagem: Python

Machine Learning: Scikit-learn, XGBoost

Análise de Dados: Pandas, NumPy, TA-Lib

API (Servidor de Inferência): Flask

Dashboard (UI): NiceGUI, Plotly

Conexão com a Exchange: PyBit

Banco de Dados: SQLite (para persistência do histórico de trades)

? Como Executar o Projeto
1. Pré-requisitos
Python 3.8 ou superior

Uma conta na exchange Bybit (real ou testnet) com chaves de API.

2. Instalação
Clone o repositório:


git clone https://github.com/seu-usuario/nome-do-repo.git
cd nome-do-repo
Crie e ative um ambiente virtual:


python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Instale as dependências:


pip install -r requirements.txt
3. Configuração
Dados Históricos: Crie uma pasta chamada data/ na raiz do projeto e adicione os arquivos CSV com os dados históricos (ex: dados_historicos_BTCUSDT_5m.csv).

Chaves de API: Abra os arquivos bot_bybit.py e painel.py e insira suas chaves de API da Bybit nos locais indicados:


API_KEY = "SUA_API_KEY"
API_SECRET = "SEU_API_SECRET"
Arquivo de Configuração: Crie um arquivo config.yaml na raiz do projeto para definir os parâmetros da estratégia. Você pode começar com este exemplo:


risk_perc: 0.01
ml_thresholds:
  BTCUSDT: 0.65
  ETHUSDT: 0.65
adx_threshold: 25

4. Execução
O sistema foi projetado para ser iniciado com um único comando, que orquestra o treinamento do modelo (se necessário), o servidor de inferência e o painel.

Treine os Modelos: (Execute apenas na primeira vez ou quando quiser retreinar)


python model_trainer.py
Inicie o Sistema Completo:
O script bot_bybit.py foi projetado para iniciar o servidor de inferência, o painel e a lógica de trading.


python bot_bybit.py
Acesse o Painel: Abra seu navegador e acesse http://localhost:8080 para monitorar o robô.

Autor
Ronaldo Neves Barbosa Neto

LinkedIn: https://www.linkedin.com/in/ronaldo-neves-barbosa-neto/