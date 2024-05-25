#include <iostream>
#include "header.hpp"
#include "function.hpp"

using namespace std;

int main(int argc, char* argv[]) {
// Verifica se o nome do arquivo foi fornecido como argumento
    if (argc < 2) {
        cout << "Uso: " << argv[0] << " <nome_arquivo>" << endl;
        return 1;
    }

    const char* nome_arquivo = argv[1];
    int n_iteracoes, n_caracteristicas, n_usuarios, n_itens;
    double alfa;
    vector<Entrada> entradas;

// Lê as entradas do arquivo
    ler_entrada_arquivo(nome_arquivo, n_iteracoes, alfa, n_caracteristicas, entradas, n_usuarios, n_itens);
    
// Inicializa as matrizes L e R com valores aleatórios
    vector<vector<double>> L(n_usuarios, vector<double>(n_caracteristicas));
    vector<vector<double>> R(n_itens, vector<double>(n_caracteristicas));
    preenche_aleatorio_LR(n_usuarios, n_itens, n_caracteristicas, L, R);

// Atualiza as matrizes L e R por várias iterações
    for (int iter = 0; iter < n_iteracoes; ++iter) {
        ActualizarMatriz(L, R, entradas, n_caracteristicas, alfa);
    }

// Recomenda itens para cada usuário e imprime os resultados
    vector<int> itens_recomendados = FuncaoRecomendarItems(L, R);
    for (int item : itens_recomendados) {
        cout << item << endl;
    }

    return 0;
}