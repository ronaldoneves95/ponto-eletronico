Sistema de Trading Algor�tmico com IA para Criptomoedas
Este reposit�rio cont�m o c�digo-fonte de um ecossistema completo para trading algor�tmico no mercado de criptomoedas, utilizando Machine Learning para prever movimentos de pre�o e executar opera��es de forma aut�noma na exchange Bybit.

? Vis�o Geral do Projeto
O sistema � composto por m�ltiplos m�dulos que trabalham em conjunto para cobrir todo o ciclo de vida de uma estrat�gia de trading quantitativo:

Engenharia de Features: Cria��o de um conjunto rico de indicadores t�cnicos a partir de dados hist�ricos.

Treinamento de Modelo: Treinamento de modelos de Machine Learning (XGBoost) especializados para cada par de criptomoedas.

Backtesting e Otimiza��o: Valida��o rigorosa da estrat�gia em dados passados e otimiza��o de hiperpar�metros para maximizar o retorno ajustado ao risco.

Execu��o em Tempo Real: Um rob� que consome as predi��es do modelo, gerencia risco e executa ordens na Bybit.

Monitoramento e Controle: Um painel web interativo para acompanhar a performance, visualizar posi��es e intervir manualmente, se necess�rio.

?? Arquitetura
O projeto � modularizado para garantir clareza, manuten��o e escalabilidade:

feature_engineering.py: M�dulo central para calcular todos os indicadores t�cnicos (EMAs, RSI, ADX, Bandas de Bollinger, etc.), garantindo consist�ncia entre o treinamento e a execu��o.

model_trainer.py: Script respons�vel por treinar um modelo XGBoost para cada ativo, criando um "artefato" (.pkl) que empacota o modelo, a lista de features e a fun��o de pr�-processamento.

inference_server.py: Um servidor de API (Flask) que carrega os modelos treinados e exp�e um endpoint /predict para fornecer previs�es em tempo real, desacoplando a IA do rob� principal.

backtest.py: Ferramenta para simular a execu��o da estrat�gia em dados hist�ricos, considerando custos realistas como taxas e slippage.

otimizador.py: Utiliza Grid Search para testar m�ltiplas combina��es de par�metros (ex: limiar de confian�a do modelo, limiar de ADX) e encontrar a configura��o mais lucrativa.

bot_bybit.py: O cora��o do sistema. Este rob� orquestra todo o processo em tempo real: busca dados, consulta a API de infer�ncia, calcula o tamanho da posi��o com base no risco, executa ordens e gerencia o Trailing Stop Loss em uma thread separada.

painel.py: Uma interface de usu�rio (NiceGUI) que fornece um dashboard completo para monitoramento, com gr�ficos de performance, visualiza��o de posi��es abertas e logs em tempo real.

?? Tecnologias Utilizadas
Linguagem: Python

Machine Learning: Scikit-learn, XGBoost

An�lise de Dados: Pandas, NumPy, TA-Lib

API (Servidor de Infer�ncia): Flask

Dashboard (UI): NiceGUI, Plotly

Conex�o com a Exchange: PyBit

Banco de Dados: SQLite (para persist�ncia do hist�rico de trades)

? Como Executar o Projeto
1. Pr�-requisitos
Python 3.8 ou superior

Uma conta na exchange Bybit (real ou testnet) com chaves de API.

2. Instala��o
Clone o reposit�rio:


git clone https://github.com/seu-usuario/nome-do-repo.git
cd nome-do-repo
Crie e ative um ambiente virtual:


python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Instale as depend�ncias:


pip install -r requirements.txt
3. Configura��o
Dados Hist�ricos: Crie uma pasta chamada data/ na raiz do projeto e adicione os arquivos CSV com os dados hist�ricos (ex: dados_historicos_BTCUSDT_5m.csv).

Chaves de API: Abra os arquivos bot_bybit.py e painel.py e insira suas chaves de API da Bybit nos locais indicados:


API_KEY = "SUA_API_KEY"
API_SECRET = "SEU_API_SECRET"
Arquivo de Configura��o: Crie um arquivo config.yaml na raiz do projeto para definir os par�metros da estrat�gia. Voc� pode come�ar com este exemplo:


risk_perc: 0.01
ml_thresholds:
  BTCUSDT: 0.65
  ETHUSDT: 0.65
adx_threshold: 25

4. Execu��o
O sistema foi projetado para ser iniciado com um �nico comando, que orquestra o treinamento do modelo (se necess�rio), o servidor de infer�ncia e o painel.

Treine os Modelos: (Execute apenas na primeira vez ou quando quiser retreinar)


python model_trainer.py
Inicie o Sistema Completo:
O script bot_bybit.py foi projetado para iniciar o servidor de infer�ncia, o painel e a l�gica de trading.


python bot_bybit.py
Acesse o Painel: Abra seu navegador e acesse http://localhost:8080 para monitorar o rob�.

Autor
Ronaldo Neves Barbosa Neto

LinkedIn: https://www.linkedin.com/in/ronaldo-neves-barbosa-neto/