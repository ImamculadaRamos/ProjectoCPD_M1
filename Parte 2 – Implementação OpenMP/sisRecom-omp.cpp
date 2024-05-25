#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>

using namespace std;

#define ALEATORIO ((double)rand() / (double)RAND_MAX)

struct Entrada {
    int linha;
    int coluna;
    double valor;
};

void ler_entrada_arquivo(const char* nome_arquivo, int* n_iteracoes, double* alpha, int* n_caracteristicas, vector<Entrada>& entradas, int* n_usuarios, int* n_itens) {
    ifstream arquivo(nome_arquivo);
    if (!arquivo.is_open()) {
        cout << "Erro ao abrir o arquivo." << endl;
        exit(1);
    }

    arquivo >> *n_iteracoes;
    arquivo >> *alpha;
    arquivo >> *n_caracteristicas;

    int n_linhas, n_colunas, nnz;
    arquivo >> n_linhas >> n_colunas >> nnz;

    *n_usuarios = n_linhas;
    *n_itens = n_colunas;

    entradas.resize(nnz);
    for (int i = 0; i < nnz; ++i) {
        arquivo >> entradas[i].linha >> entradas[i].coluna >> entradas[i].valor;
    }

    arquivo.close();
}

void preenche_aleatorio_LR(int nU, int nI, int nF, vector<vector<double>>& L, vector<vector<double>>& R) {
    srand(0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nU; i++) {
        for (int j = 0; j < nF; j++) {
            L[i][j] = ALEATORIO / (double)nF;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nF; i++) {
        for (int j = 0; j < nI; j++) {
            R[i][j] = ALEATORIO / (double)nF;
        }
    }
}

void atualizar_matrizes_gradiente(vector<vector<double>>& L, vector<vector<double>>& R, const vector<Entrada>& entradas, int n_caracteristicas, double alpha) {
    int n_usuarios = L.size();
    int n_itens = R[0].size();
    vector<vector<double>> B(n_usuarios, vector<double>(n_itens, 0.0));

    #pragma omp parallel for
    for (int idx = 0; idx < entradas.size(); ++idx) {
        int i = entradas[idx].linha;
        int j = entradas[idx].coluna;
        for (int k = 0; k < n_caracteristicas; ++k) {
            #pragma omp atomic
            B[i][j] += L[i][k] * R[k][j];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n_usuarios; ++i) {
        for (int k = 0; k < n_caracteristicas; ++k) {
            double gradiente_L = 0.0;
            for (const auto& entrada : entradas) {
                if (entrada.linha == i) {
                    int j = entrada.coluna;
                    gradiente_L += 2 * (entrada.valor - B[i][j]) * R[k][j];
                }
            }
            #pragma omp atomic
            L[i][k] += alpha * gradiente_L;
        }
    }

    #pragma omp parallel for
    for (int j = 0; j < n_itens; ++j) {
        for (int k = 0; k < n_caracteristicas; ++k) {
            double gradiente_R = 0.0;
            for (const auto& entrada : entradas) {
                if (entrada.coluna == j) {
                    int i = entrada.linha;
                    gradiente_R += 2 * (entrada.valor - B[i][j]) * L[i][k];
                }
            }
            #pragma omp atomic
            R[k][j] += alpha * gradiente_R;
        }
    }
}

vector<int> recomendar_itens_usuario(const vector<vector<double>>& L, const vector<vector<double>>& R) {
    int n_usuarios = L.size();
    int n_itens = R[0].size();
    vector<int> itens_recomendados(n_usuarios);

    #pragma omp parallel for
    for (int i = 0; i < n_usuarios; ++i) {
        double max_valor = -1e9;
        int melhor_item = -1;

        for (int j = 0; j < n_itens; ++j) {
            double valor_estimado = 0.0;
            for (int k = 0; k < L[i].size(); ++k) {
                valor_estimado += L[i][k] * R[k][j];
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

double calcular_erro_quadratico(const vector<vector<double>>& L, const vector<vector<double>>& R, const vector<Entrada>& entradas) {
    double erro_quadratico_total = 0.0;

    #pragma omp parallel for reduction(+:erro_quadratico_total)
    for (int idx = 0; idx < entradas.size(); ++idx) {
        int i = entradas[idx].linha;
        int j = entradas[idx].coluna;
        double valor_real = entradas[idx].valor;
        double valor_estimado = 0.0;

        for (int k = 0; k < L[i].size(); ++k) {
            valor_estimado += L[i][k] * R[k][j];
        }

        erro_quadratico_total += pow(valor_real - valor_estimado, 2);
    }

    return erro_quadratico_total;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Uso: " << argv[0] << " <arquivo_entrada>" << endl;
        return 1;
    }

    const char* nome_arquivo = argv[1];
    int n_iteracoes, n_caracteristicas, n_usuarios, n_itens;
    double alpha;
    vector<Entrada> entradas;

    ler_entrada_arquivo(nome_arquivo, &n_iteracoes, &alpha, &n_caracteristicas, entradas, &n_usuarios, &n_itens);
    
    vector<vector<double>> L(n_usuarios, vector<double>(n_caracteristicas));
    vector<vector<double>> R(n_caracteristicas, vector<double>(n_itens));

    preenche_aleatorio_LR(n_usuarios, n_itens, n_caracteristicas, L, R);

    for (int iter = 0; iter < n_iteracoes; ++iter) {
        atualizar_matrizes_gradiente(L, R, entradas, n_caracteristicas, alpha);
    }

    vector<int> itens_recomendados = recomendar_itens_usuario(L, R);
    for (int i = 0; i < n_usuarios; ++i) {
        cout << itens_recomendados[i] << endl;
    }

    double erro_quadratico = calcular_erro_quadratico(L, R, entradas);
    cout << "Erro Quadrático Total: " << erro_quadratico << endl;

    return 0;
}
