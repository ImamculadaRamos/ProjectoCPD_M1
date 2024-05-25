#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "header.hpp"

#define ALEATORIO ((double)random() / (double)RAND_MAX)

using namespace std;

// Estrutura para armazenar uma entrada do arquivo
struct Entrada {
    int linha;
    int coluna;
    double valor;
};

// Função para ler as entradas do arquivo
void ler_entrada_arquivo(const char* nome_arquivo, int& n_iteracoes, double& alfa, int& n_caracteristicas, vector<Entrada>& entradas, int& n_usuarios, int& n_itens) {
    
    ifstream arquivo(nome_arquivo);
    if (!arquivo.is_open()) {
        cout << "Erro ao abrir o arquivo." << endl;
        exit(1);
    }

    // Lê os parâmetros iniciais
    arquivo >> n_iteracoes;
    arquivo >> alfa;
    arquivo >> n_caracteristicas;

    // Lê o número de usuários, itens e entradas não-nulas
    int dados_Ficheiro;
    arquivo >> n_usuarios >> n_itens >> dados_Ficheiro;

    // Lê as entradas do arquivo
    entradas.resize(dados_Ficheiro);
    for (int i = 0; i < dados_Ficheiro; ++i) {
        arquivo >> entradas[i].linha >> entradas[i].coluna >> entradas[i].valor;
    }

    arquivo.close();
}

// Função para preencher as matrizes L e R com valores aleatórios
void preenche_aleatorio_LR(int nU, int nI, int nF, vector<vector<double>>& L, vector<vector<double>>& R) {
    srandom(0); // Inicializa a semente do gerador de números aleatórios

// Preenche a matriz L com valores aleatórios
    for (int i = 0; i < nU; i++) {
        L[i].resize(nF);
        for (int j = 0; j < nF; j++) {
            L[i][j] = ALEATORIO / (double)nF;
        }
    }

// Preenche a matriz R com valores aleatórios
    for (int i = 0; i < nF; i++) {
        R[i].resize(nI);
        for (int j = 0; j < nI; j++) {
            R[i][j] = ALEATORIO / (double)nF;
        }
    }
}


// Função para atualizar as matrizes L e R com base nas entradas e no gradiente
void ActualizarMatriz(vector<vector<double>>& L, vector<vector<double>>& R, const vector<Entrada>& entradas, int n_caracteristicas, double alfa) {
    int n_usuarios = L.size();
    int n_itens = R.size();
    vector<vector<double>> B(n_usuarios, vector<double>(n_itens, 0.0));

// Criar e preencher matriz esparsa para as entradas não-nulas
    vector<vector<double>> entradas_esparsas(n_usuarios, vector<double>(n_itens, 0.0));
    for (const auto& entrada : entradas) {
        entradas_esparsas[entrada.linha][entrada.coluna] = entrada.valor;
    }

// Atualizar L e R usando as entradas esparsas
    for (int i = 0; i < n_usuarios; ++i) {
        for (int j = 0; j < n_itens; ++j) {
            if (entradas_esparsas[i][j] != 0.0) {
                for (int k = 0; k < n_caracteristicas; ++k) {
                    B[i][j] += L[i][k] * R[j][k];
                }
            }
        }
    }

// Atualiza a matriz L
    for (int i = 0; i < n_usuarios; ++i) {
        for (int k = 0; k < n_caracteristicas; ++k) {
            double gradiente_L = 0.0;
            for (int j = 0; j < n_itens; ++j) {
                if (entradas_esparsas[i][j] != 0.0) {
                    gradiente_L += 2 * (entradas_esparsas[i][j] - B[i][j]) * R[j][k];
                }
            }
            L[i][k] += alfa * gradiente_L;
        }
    }
    
// Atualiza a matriz R
    for (int j = 0; j < n_itens; ++j) {
        for (int k = 0; k < n_caracteristicas; ++k) {
            double gradiente_R = 0.0;
            for (int i = 0; i < n_usuarios; ++i) {
                if (entradas_esparsas[i][j] != 0.0) {
                    gradiente_R += 2 * (entradas_esparsas[i][j] - B[i][j]) * L[i][k];
                }
            }
            R[j][k] += alfa * gradiente_R;
        }
    }
}


// Função para recomendar itens para cada usuário
vector<int> FuncaoRecomendarItems(const vector<vector<double>>& L, const vector<vector<double>>& R) {
    int n_usuarios = L.size();
    int n_itens = R.size();
    vector<int> itens_recomendados(n_usuarios, -1);

    for (int i = 0; i < n_usuarios; ++i) {
        double max_valor = -1e9;
        int melhor_item = -1;

// Calcula o valor estimado para cada item para o usuário i
        for (int j = 0; j < n_itens; ++j) {
            double valor_estimado = 0.0;
            for (int k = 0; k < L[i].size(); ++k) {
                valor_estimado += L[i][k] * R[j][k];
            }
            if (valor_estimado > max_valor) {
                max_valor = valor_estimado;
                melhor_item = j;
            }
        }
        itens_recomendados[i] = melhor_item;
    }

    return itens_recomendados;
}