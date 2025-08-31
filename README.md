Gerenciador de Ponto Eletrônico
por Ronaldo Neves B Neto

📖 Sobre o Projeto
O Gerenciador de Ponto é um sistema web completo, desenvolvido em Python com o framework Flask, para controlar e gerenciar o registro de ponto de funcionários. A aplicação foi criada para ser simples e eficiente, permitindo o registro de ponto de duas formas: manual ou através de um leitor de código de barras, ideal para portarias e recepções com alto fluxo de pessoas.

O sistema conta com três níveis de acesso (Administrador, Segurança e Comum), relatórios detalhados em PDF, dashboards com gráficos e um sistema de backup do banco de dados.

✨ Funcionalidades Principais
O sistema é dividido em três perfis de usuário, cada um com suas permissões específicas:

1. Perfil de Administrador (admin)
O administrador tem controle total sobre o sistema.

Dashboard Intuitivo: Exibe gráficos com a distribuição de funcionários por tipo de perfil e a quantidade de pontos batidos no dia.

Gerenciamento de Usuários: Permite cadastrar, visualizar, editar e excluir funcionários. É possível filtrar usuários por nome, matrícula, tipo e data de cadastro.

Edição de Pontos: Corrige registros de ponto de qualquer funcionário, ajustando horários de entrada e saída quando necessário.

Geração de Códigos de Barras: Cria e exibe o código de barras associado à matrícula de cada funcionário, que pode ser impresso em crachás.

Backup do Banco de Dados: Com um clique, o sistema gera um arquivo .zip contendo o backup do banco de dados (SQLite), o envia para um e-mail pré-configurado e o disponibiliza para download.

Relatórios Avançados em PDF: Gera relatórios de ponto detalhados e personalizados.

Como Funcionam os Relatórios?
Esta é uma das funcionalidades mais poderosas do sistema. O administrador pode filtrar por:

Período: Hoje, semanal, mensal ou um intervalo de datas personalizado.

Funcionário: Seleciona um usuário específico.

Carga Horária: Define a jornada de trabalho padrão (4, 7 ou 8 horas diárias).

Com base nesses filtros, o sistema gera um PDF que calcula:

Total de Horas Trabalhadas no Dia: Soma os intervalos entre as entradas e saídas ((saida1 - entrada1) + (saida2 - entrada2)).

Horas Extras / Horas Devidas (Extra/Devedor): Compara o total de horas trabalhadas no dia com a Carga Horária padrão.

Horas Positivas (Extras): Se o funcionário trabalhou mais que o padrão, a diferença é registrada como hora extra.

Horas Negativas (Devendo): Se trabalhou menos, a diferença é registrada como hora a dever, exibida em vermelho para fácil identificação.

Saldo Final: Ao final do relatório, é apresentado o balanço total de horas do período, mostrando se o funcionário tem um saldo positivo (horas extras) ou negativo (horas a compensar).

2. Perfil de Segurança (seguranca)
Este perfil é focado exclusivamente no registro dos pontos.

Página de Registro de Ponto: É a única tela acessível. O segurança pode:

Bater o ponto com leitor: O cursor fica posicionado no campo de busca. Ao escanear o código de barras do funcionário, o sistema captura a matrícula e registra o ponto automaticamente.

Bater o ponto manualmente: Caso o funcionário esteja sem o crachá, o segurança pode buscar pelo nome ou matrícula e registrar o ponto.

Visualização em Tempo Real: A página exibe uma tabela com os últimos pontos registrados no dia, que é atualizada a cada novo registro.

3. Perfil Comum (comum)
O perfil do funcionário, para consulta dos próprios registros.

Dashboard Pessoal: Ao fazer login, o funcionário visualiza uma tabela com seus registros de ponto mais recentes e o total de horas trabalhadas em cada dia.

Aviso de Contato: Uma seção de avisos o orienta a procurar o RH ou seu gestor em caso de inconsistências.

🛠️ Tecnologias Utilizadas
Este projeto foi construído utilizando as seguintes tecnologias e bibliotecas:

Backend:

Python 3

Flask: Micro-framework web para criar as rotas, a lógica e servir as páginas.

SQLite: Banco de dados relacional leve e baseado em arquivo.

Frontend:

HTML5

CSS3

JavaScript: Para interatividade no lado do cliente (eventos, requisições AJAX).

Bibliotecas Python Notáveis:

werkzeug: Para hashing e segurança de senhas.

python-barcode: Para a geração das imagens de código de barras.

reportlab: Para a criação dinâmica dos relatórios em PDF.

smtplib: Para o envio de e-mails com o backup do banco de dados.

🚀 Como Executar o Projeto Localmente
Siga os passos abaixo para rodar a aplicação na sua máquina.

Pré-requisitos
Python 3.x instalado.

Git instalado.

Passos
Clone o repositório:

git clone https://github.com/ronaldoneves95/ponto-eletronico/.git
Navegue até o diretório do projeto:

cd Gerenciador-de-Ponto
Crie e ative um ambiente virtual (recomendado):

# Para Windows
python -m venv venv
venv\Scripts\activate

# Para macOS/Linux
python3 -m venv venv
source venv/bin/activate
Instale as dependências:
O arquivo requirements.txt contém todas as bibliotecas necessárias.

pip install -r "Ponto Cod Barras/requirements.txt"
Execute a aplicação:

python "Ponto Cod Barras/app.py"
Acesse o sistema:
Abra seu navegador e acesse http://127.0.0.1:5000.

Primeiros Passos
O primeiro usuário precisa ser cadastrado como admin. Na tela de cadastro, selecione o tipo "Admin" e utilize o token de administrador definido no arquivo app.py (o padrão é @ssjjti).

Após o login como admin, você poderá cadastrar os outros usuários.

🤝 Como Contribuir
Contribuições são bem-vindas! Se você tem ideias para novas funcionalidades ou encontrou algum bug, siga os passos:

Faça um fork deste repositório.

Crie uma nova branch para sua feature (git checkout -b minha-feature).

Faça o commit das suas alterações (git commit -m 'Adiciona nova funcionalidade').

Envie para a sua branch (git push origin minha-feature).

Abra um Pull Request.

📄 Licença
Este projeto está licenciado sob a Licença MIT.
