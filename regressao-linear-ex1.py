"""
@file regressao-linear-ex1.py
@brief Exercise 2 - Linear Regression implementation with visualization.

This script performs the following tasks:
1. Runs a warm-up exercise.
2. Loads and plots training data.
3. Implements cost function and gradient descent.
4. Predicts values for new inputs.
5. Visualizes the cost function surface and contour.

@author Teacher Thales Levi Azevedo Valente
@subject Foundations of Neural Networks
@course Computer Engineering
@university Federal University of Maranhão
@date 2025
"""


import numpy as np
import matplotlib.pyplot as plt
import os

from Functions.warm_up_exercises import warm_up_exercise1, warm_up_exercise2, warm_up_exercise3, warm_up_exercise4
from Functions.warm_up_exercises import warm_up_exercise5, warm_up_exercise6, warm_up_exercise7
from Functions.plot_data import plot_data
from Functions.compute_cost import compute_cost
from Functions.gradient_descent import gradient_descent

def experimento_taxas_aprendizado(x_aug, y, iterations):# código do experimento 1 aqui
    # === EXPERIMENTO 1: Comparação de Taxas de Aprendizado === fiz isso daqui só pra não ficar tão grande o main
    # Esse experimento compara o desempenho do algoritmo de descida do gradiente
    print("\n📌 Experimento 1: Comparando diferentes taxas de aprendizado (α)")

    # Solicita ao usuário três valores de alpha separados por vírgula
    alphas_input = input("Digite três valores para a taxa de aprendizado (α), separados por vírgula (ex: 0.001, 0.01, 0.1): ")

    # Converte a string digitada para uma lista de floats
    alphas = [float(val.strip()) for val in alphas_input.split(",") if val.strip()]

    # Cores para os gráficos (até 6 alphas suportadas)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # Inicializa theta como vetor de zeros, fixo para todas as comparações
    theta_init = np.zeros(2)

    # Cria gráfico de convergência
    plt.figure(figsize=(8, 5))

    # Executa gradient descent para cada alpha fornecido
    for i, alpha_val in enumerate(alphas):
        _, J_hist, _ = gradient_descent(x_aug, y, theta_init.copy(), alpha_val, iterations)
        plt.plot(np.arange(1, iterations + 1), J_hist, colors[i % len(colors)], label=f'α = {alpha_val}')

    # Ajustes de visualização do gráfico
    plt.xlabel('Iterações')
    plt.ylabel('Custo J(θ)')
    plt.title('Convergência da Função de Custo para Diferentes Taxas de Aprendizado')
    plt.legend()
    plt.grid(True)

    # Salva o gráfico como imagem
    plt.savefig("Figures/experimento_taxa_aprendizado.png", dpi=300)
    plt.show()
def experimento_inicializacoes(x_aug, y, iterations, theta0_vals, theta1_vals, j_vals):# código do experimento 2 aqui
    # === EXPERIMENTO 2: Comparação de Inicializações de Pesos (θ) === Mesma coisa do outro, só que com inicializações. Fiz assim mais pra não ficar tão grande o main, e tbm acho que ficou livre pra gente escolher né?
    # Esse experimento compara o desempenho do algoritmo de descida do gradiente
    print("\n📌 Experimento 2: Comparando diferentes inicializações dos pesos θ")

    # Lista para armazenar as inicializações fornecidas
    theta_inputs = []

    # Solicita 3 inicializações manuais (fixas) ao usuário
    print("Digite 3 inicializações fixas para θ, no formato 'θ0 θ1' (ex: 0 0 ou -5 5):")
    for i in range(1, 4):
        entrada = input(f"Inicialização fixa {i}: ")
        try:
            valores = [float(v.strip()) for v in entrada.split()]
            if len(valores) == 2:
                theta_inputs.append(np.array(valores))
            else:
                print("❗ Formato inválido. Usando [0, 0] por padrão.")
                theta_inputs.append(np.array([0.0, 0.0]))
        except ValueError:
            print("❗ Entrada inválida. Usando [0, 0] por padrão.")
            theta_inputs.append(np.array([0.0, 0.0]))

    # Adiciona 3 inicializações aleatórias (simulando casos reais de random init)
    for _ in range(3):
        theta_inputs.append(np.random.randn(2))  # distribuição normal

    # Cria o gráfico de contorno da função de custo
    plt.figure(figsize=(8, 6))
    plt.contour(theta0_vals, theta1_vals, j_vals, levels=np.logspace(-2, 3, 20))
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.title('Trajetórias do Gradiente para Diferentes Inicializações de θ')
    plt.grid(True)

    # Traça a trajetória do gradiente para cada θ inicial
    for i, init_theta in enumerate(theta_inputs):
        _, _, th_hist = gradient_descent(x_aug, y, init_theta.copy(), alpha=0.01, num_iters=iterations)
        label = f'init {i+1}: {init_theta.round(2)}'
        plt.plot(th_hist[:, 0], th_hist[:, 1], marker='o', markersize=3, label=label)

    # Finaliza e salva o gráfico
    plt.legend(fontsize=8)
    plt.savefig("Figures/experimento_inicializacao_pesos.png", dpi=300)
    plt.show()
def main():
    """
    @brief Executa todos os passos do exercício de regressão linear.

    Esta função serve como ponto de partida para o exercício completo de regressão linear.
    Ela executa uma série de etapas fundamentais, utilizadas como base para o aprendizado
    de modelos supervisionados em redes neurais.

    As principais etapas executadas são:
      1. Executa o exercício de aquecimento (warm-up), imprimindo uma matriz identidade 5x5.
      2. Carrega e plota os dados de treinamento de uma regressão linear simples.
      3. Calcula o custo com diferentes valores de theta usando a função de custo J(θ).
      4. Executa o algoritmo de descida do gradiente para minimizar a função de custo.
      5. Plota a linha de regressão ajustada sobre os dados originais.
      6. Realiza previsões para valores populacionais de 35.000 e 70.000.
      7. Visualiza a função de custo J(θ₀, θ₁) em gráfico de superfície 3D e gráfico de contorno.

    @instructions
    - Os alunos devem garantir que todas as funções auxiliares estejam implementadas corretamente:
        * warm_up_exercise()
        * plot_data()
        * compute_cost()
        * gradient_descent()
    - Todas as funções devem seguir padrão PEP8 e possuir docstrings no formato Doxygen.
    - O script deve ser executado a partir do `main()`.

    @note
    O dataset de entrada `ex1data1.txt` deve estar no mesmo diretório Data.
    A estrutura esperada dos dados é: [population, profit].

    @return None
    """
    # Configurações do matplotlib para salvar figuras em alta qualidade
    # Garante que a pasta de figuras existe
    os.makedirs("Figures", exist_ok=True)

    print('Executando o exercício de aquecimento (warm_up_exercise)...')
    print('Matriz identidade 5x5:')
    # Executa a função de aquecimento
    # Essa função deve retornar uma matriz identidade 5x5
    # representada como um array do NumPy.
    # A função está definida em Functions/warm_up_exercise.py
    # e foi importada no início deste arquivo.
    print('Executando os exercícios de aquecimento...')

    # Exercício 1: Matriz identidade 5x5
    print('\nExercício 1: Matriz identidade 5x5') 
    print(warm_up_exercise1()) # Esperado: [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]

    # Exercício 2: Vetor coluna de 1s
    print('\nExercício 2: Vetor de 1s (m=3)')
    print(warm_up_exercise2(3)) # Esperado: [[1], [1], [1]]

    # Exercício 3: Adiciona bias ao vetor x
    print('\nExercício 3: Adiciona coluna de 1s ao vetor [2, 4, 6]')
    x_ex3 = np.array([2, 4, 6])
    print(warm_up_exercise3(x_ex3)) # Esperado: [[1, 2], [1, 4], [1, 6]]

    # Exercício 4: Produto matricial X @ theta
    print('\nExercício 4: Produto X @ theta')
    X_ex4 = warm_up_exercise3(x_ex3)
    theta_ex4 = np.array([1, 2])
    print(warm_up_exercise4(X_ex4, theta_ex4))  # Esperado: [5, 9, 13]

    # Exercício 5: Erros quadráticos
    print('\nExercício 5: Erros quadráticos entre predições e y')
    preds = warm_up_exercise4(X_ex4, theta_ex4)
    y_ex5 = np.array([5, 9, 13])
    print(warm_up_exercise5(preds, y_ex5))  # Esperado: [0, 0, 0]

    # Exercício 6: Custo médio a partir dos erros
    print('\nExercício 6: Custo médio')
    errors_ex6 = warm_up_exercise5(preds, y_ex5)
    print(warm_up_exercise6(errors_ex6))  # Esperado: 0.0

    # Exercício 7: Custo médio com base em X, y e theta
    print('\nExercício 7: Cálculo do custo médio completo')
    print(warm_up_exercise7(X_ex4, y_ex5, theta_ex4))  # Esperado: 0.0


    input("Programa pausado. Pressione Enter para continuar...")

    print('Plotando os dados...')
    # Carrega os dados de treinamento a partir do arquivo ex1data1.txt
    # O arquivo contém duas colunas: a primeira com a população da cidade
    # (em dezenas de milhar) e a segunda com o lucro (em dezenas de mil dólares).
    # Os dados são carregados usando a função np.loadtxt do NumPy.
    # A função np.loadtxt lê os dados do arquivo e os armazena em um array NumPy.
    data = np.loadtxt('Data/ex1data1.txt', delimiter=',')
    # Separa os dados em duas variáveis: x e y
    # x contém a população da cidade (em dezenas de milhar)
    # y contém o lucro (em dezenas de mil dólares)
    # A primeira coluna de data é a população (x), a feature
    # que será usada para prever o lucro.
    x = data[:, 0] 
    # A segunda coluna de data é o lucro (y), a label ou target
    y = data[:, 1]
    # A variável y contém os valores reais de lucro correspondentes
    # Agora, obtemos o número de exemplos de treinamento (m)
    m = len(y)

    # Plotagem dos dados
    # Utiliza a função plot_data para exibir os pontos (x, y) em um gráfico 2D.
    # A função está definida em Functions/plot_data.py
    # e foi importada no início do arquivo.
    plot_data(x,y) #tinha um espaço meio estranho aqui, mas não sei se era só impressão
    # A função plot_data recebe como parâmetros os dados de entrada (x) e os valores de saída (y).

    input("Programa pausado. Pressione Enter para continuar...")

    # Preparação dos dados para o algoritmo de descida do gradiente
    # Adiciona uma coluna de 1s à matriz x para representar o termo de interceptação (bias).
    # Isso é feito com np.column_stack, combinando uma coluna de 1s com os valores de x.
    # A nova matriz x_aug terá duas colunas: a primeira com 1s e a segunda com os valores originais de x.
    x_aug = np.column_stack((np.ones(m), x))

    # Inicialização de theta como um vetor nulo (vetor de zeros)
    # Inicializa o vetor de parâmetros theta como um vetor nulo com 2 elementos (theta[0] e theta[1]).
    # O primeiro elemento representa o intercepto (bias) e o segundo o coeficiente angular (inclinação).
    # Esse vetor será ajustado durante a execução do algoritmo de descida do gradiente.
    theta =  np.zeros(2)

    # Parâmetros da descida do gradiente
    # Define o número de iterações e a taxa de aprendizado (alpha)
    # O número de iterações determina quantas vezes os parâmetros serão atualizados.
    iterations = 1500

    # A taxa de aprendizado (alpha) controla o tamanho do passo dado em cada iteração do algoritmo de descida do gradiente.
    # Um alpha muito grande pode fazer o algoritmo divergir, enquanto um muito pequeno pode torná-lo lento.
    # Aqui, alpha é definido como 0.01, um valor comumente usado em problemas de regressão linear.
    # Você pode experimentar outros valores para ver como o algoritmo se comporta.
    alpha = 0.01

    print('\nTestando a função de custo...')
    # Utiliza a função compute_cost para calcular o custo com os parâmetros iniciais (theta = [0, 0]).
    # Essa função mede o quão bem os parâmetros atuais se ajustam aos dados de treinamento.
    # Ela está definida em Functions/compute_cost.py e foi importada anteriormente.
    # Os parâmetros de entrada são a matriz x_aug (com 1s e valores de x), o vetor y (lucro) e o vetor theta (parâmetros).
    cost = compute_cost(x_aug, y, theta)
    print(f'Com theta = [0 ; 0]\nCusto calculado = {cost:.2f}')
    print('Valor esperado para o custo (aproximadamente): 32.07')

    # Testando a função de custo com outro valor de theta
    # Aqui, testamos a função de custo com um valor diferente de theta ([-1, 2]).
    # Isso nos permite verificar se a função de custo está funcionando corretamente.
    # O valor de theta = [-1, 2] é um exemplo e não representa o ajuste ideal.
    cost = compute_cost(x_aug, y, np.array([-1, 2]))
    print(f'\nCom theta = [-1 ; 2]\nCusto calculado = {cost:.2f}')
    print('Valor esperado para o custo (aproximadamente): 54.24')

    input("Programa pausado. Pressione Enter para continuar...")

    print('\nExecutando a descida do gradiente...')
    # Executa o algoritmo de descida do gradiente para encontrar os parâmetros ótimos (theta).
    # A função gradient_descent é definida em Functions/gradient_descent.py
    # e foi importada anteriormente.
    # Ela recebe como parâmetros a matriz x_aug (com 1s e valores de x), o vetor y (lucro),

    # Após os testes, inicializamos os parâmetros theta com valores diferentes de zero.
    # theta = [8.5, 4.0] é o ponto de partida padrão. Foi estabelecido empiricamente ao olhar os gráficos.
    # Você pode experimentar outros valores para ver como o algoritmo se comporta.
    theta = np.array([8.5, 4.0])
    # o vetor theta, a taxa de aprendizado (alpha) e o número de iterações.
    # A função retorna os parâmetros ajustados (theta), o histórico de custos (J_history) e o histórico de theta (theta_history).
    # O histórico de custos é usado para visualizar a convergência do algoritmo.
    # O histórico de theta é usado para visualizar a trajetória do gradiente na superfície da função de custo.
    theta, J_history, theta_history = gradient_descent(x_aug, y, theta, alpha, iterations)

    print('Parâmetros theta encontrados pela descida do gradiente:')
    print(theta)
    print('Valores esperados para theta (aproximadamente):')
    print(' -3.6303\n  1.1664')

    # 1. Gráfico da convergência da função de custo
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, iterations + 1), J_history, 'b-', linewidth=2)
    plt.xlabel('Iteração')
    plt.ylabel('Custo J(θ)')
    plt.title('Convergência da Descida do Gradiente')
    plt.savefig("Figures/convergencia_custo.png", dpi=300, bbox_inches='tight')
    plt.savefig("Figures/convergencia_custo.svg", format='svg', bbox_inches='tight')
    plt.grid(True)
    plt.show()

    # 2. Gráfico do Ajuste da Regressão Linear
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'rx', markersize=5, label='Dados de treino')
    plt.plot(x, x_aug @ theta, 'b-', linewidth=2, label='Regressão linear')
    plt.xlabel('População da cidade (em dezenas de mil)')
    plt.ylabel('Lucro (em dezenas de mil dólares)')
    plt.title('Ajuste da Regressão Linear')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Previsões usando theta ajustado
    predict1 = np.array([1, 3.5]) @ theta
    predict2 = np.array([1, 7.0]) @ theta
    print(f'\nPara população = 35.000, lucro previsto = ${predict1 * 10000:.2f}')
    print(f'Para população = 70.000, lucro previsto = ${predict2 * 10000:.2f}')
    input("Programa pausado. Pressione Enter para continuar...")

    # 3. Gráfico de superfície 3D da função de custo J(θ₀, θ₁)
    print('Visualizando a função J(theta_0, theta_1) – superfície 3D...')
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    j_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i, t0 in enumerate(theta0_vals):
        for j, t1 in enumerate(theta1_vals):
            j_vals[i, j] = compute_cost(x_aug, y, np.array([t0, t1]))
    j_vals = j_vals.T

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    t0_mesh, t1_mesh = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(t0_mesh, t1_mesh, j_vals, cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel('Custo')
    plt.title('Superfície da Função de Custo')
    plt.show()

    # 4. Gráfico de contorno da função de custo
    print('Visualizando a função J(theta_0, theta_1) – contorno...')
    plt.figure(figsize=(8, 5))
    plt.contour(theta0_vals, theta1_vals, j_vals, levels=np.logspace(-2, 3, 20))
    plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.title('Contorno da Função de Custo')
    plt.grid(True)
    plt.show()

    # 5) Contorno da função de custo + trajetória do gradiente
    plt.figure(figsize=(8, 5))
    # desenha as linhas de contorno
    cs = plt.contour(theta0_vals, theta1_vals, j_vals,
                     levels=np.logspace(-2, 3, 20))
    plt.clabel(cs, inline=1, fontsize=8)  # mostra valores de custo nas linhas

    # sobrepõe a trajetória dos thetas
    plt.plot(theta_history[:, 0], theta_history[:, 1],
             'r.-', markersize=6, label='Trajetória do gradiente')

    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.title('Contorno da Função de Custo com Trajetória')
    plt.legend()
    plt.grid(True)
    plt.savefig("Figures/contorno_trajetoria.png", dpi=300, bbox_inches='tight')
    plt.savefig("Figures/contorno_trajetoria.svg", format='svg', bbox_inches='tight')
    plt.show()

    # 7) Superfície da função de custo com trajetória 3D melhor visualizada
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 7.1 Plota a superfície semitransparente
    surf = ax.plot_surface(
        t0_mesh, t1_mesh, j_vals,
        cmap='viridis',
        edgecolor='none',
        alpha=0.6       # deixa a superfície semitransparente
    )

    # 7.2 Ajusta ângulo de visão
    ax.view_init(elev=18, azim=-18, roll=-5)

    # 7.3 Trajetória do gradiente em linha vermelha grossa
    costs = np.concatenate(
        ([compute_cost(x_aug, y, theta_history[0])], J_history)
    )

    # Inserindo a trajetória 3D do gradiente
    # theta_history: shape (iter+1, 2), J_history: shape (iter,)
    ax.plot(
        theta_history[:, 0], 
        theta_history[:, 1], costs,
        color='red',
        linewidth=3,
        marker='o',
        markersize=4,
        label='Trajetória do gradiente'
    )

    # 7.4 Destacar ponto inicial e final
    ax.scatter(
        theta_history[0, 0], theta_history[0, 1], costs[0],
        color='blue', s=50, label='Início'
    )
    ax.scatter(
        theta_history[-1, 0], theta_history[-1, 1], costs[-1],
        color='green', s=50, label='Convergência'
    )
    
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel('Custo')
    plt.title('Superfície da Função de Custo com Trajetória 3D')
    ax.legend()
    plt.savefig("Figures/superficie_trajetoria.png", dpi=300, bbox_inches='tight')
    plt.savefig("Figures/superficie_trajetoria.svg", format='svg', bbox_inches='tight')
    plt.show()

    #Experimentos aqui:
    # === EXPERIMENTOS COMPARATIVOS INTERATIVOS ===

    # 1. Comparando diferentes taxas de aprendizado (α)
    # ------------------------------------------------
    # Código completo para solicitar 3 valores de α via input()
    # Executar gradient descent para cada α
    # Plotar gráfico comparativo de convergência
    # Salvar e mostrar o gráfico

    # 2. Comparando diferentes inicializações dos pesos (θ)
    # -----------------------------------------------------
    # Solicitar 3 inicializações fixas via input()
    # Adicionar 3 aleatórias
    # Executar gradient descent com cada inicialização
    # Plotar gráfico de contorno com trajetórias
    # Salvar e mostrar o gráfico
    experimento_taxas_aprendizado(x_aug, y, iterations)
    experimento_inicializacoes(x_aug, y, iterations, theta0_vals, theta1_vals, j_vals)

if __name__ == '__main__':
    main()
