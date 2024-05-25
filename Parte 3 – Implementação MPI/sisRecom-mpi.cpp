#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mpi.h>

using namespace std;

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

    arquivo >> *n_iteracoes >> *alpha >> *n_caracteristicas;

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
    L.resize(nU, vector<double>(nF));
    R.resize(nF, vector<double>(nI));

    for (int i = 0; i < nU; i++) {
        for (int j = 0; j < nF; j++) {
            L[i][j] = (double)rand() / RAND_MAX / nF;
        }
    }

    for (int i = 0; i < nF; i++) {
        for (int j = 0; j < nI; j++) {
            R[i][j] = (double)rand() / RAND_MAX / nF;
        }
    }
}

void atualizar_matrizes_gradiente(vector<vector<double>>& L, vector<vector<double>>& R, const vector<Entrada>& entradas, int n_caracteristicas, double alpha) {
    vector<vector<double>> B(L.size(), vector<double>(R[0].size(), 0.0));

    for (const auto& entrada : entradas) {
        int i = entrada.linha;
        int j = entrada.coluna;
        for (int k = 0; k < n_caracteristicas; ++k) {
            B[i][j] += L[i][k] * R[k][j];
        }
    }

    for (const auto& entrada : entradas) {
        int i = entrada.linha;
        int j = entrada.coluna;
        for (int k = 0; k < n_caracteristicas; ++k) {
            L[i][k] += alpha * (2 * (entrada.valor - B[i][j]) * R[k][j]);
            R[k][j] += alpha * (2 * (entrada.valor - B[i][j]) * L[i][k]);
        }
    }
}

double calcular_erro_quadratico(const vector<vector<double>>& L, const vector<vector<double>>& R, const vector<Entrada>& entradas) {
    double erro_quadratico_total = 0.0;

    for (const auto& entrada : entradas) {
        int i = entrada.linha;
        int j = entrada.coluna;
        double valor_real = entrada.valor;
        double valor_estimado = 0.0;

        for (size_t k = 0; k < L[i].size(); ++k) {
            valor_estimado += L[i][k] * R[k][j];
        }

        erro_quadratico_total += pow(valor_real - valor_estimado, 2);
    }

    return erro_quadratico_total;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int n_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2) {
        if (rank == 0) {
            cout << "Uso: " << argv[0] << " <arquivo_entrada>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    const char* nome_arquivo = argv[1];
    int n_iteracoes, n_caracteristicas, n_usuarios, n_itens;
    double alpha;
    vector<Entrada> entradas;

    if (rank == 0) {
        ler_entrada_arquivo(nome_arquivo, &n_iteracoes, &alpha, &n_caracteristicas, entradas, &n_usuarios, &n_itens);
    }

    MPI_Bcast(&n_iteracoes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_caracteristicas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_usuarios, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_itens, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<vector<double>> L, R;
    if (rank == 0) {
        preenche_aleatorio_LR(n_usuarios, n_itens, n_caracteristicas, L, R);
    }

    // Distribuir L e R para todos os processos
    // Apenas uma abordagem simplificada aqui, pode ser otimizada
    vector<double> L_buffer(n_usuarios * n_caracteristicas);
    vector<double> R_buffer(n_itens * n_caracteristicas);

    if (rank == 0) {
        for (int i = 0; i < n_usuarios; ++i) {
            for (int j = 0; j < n_caracteristicas; ++j) {
                L_buffer[i * n_caracteristicas + j] = L[i][j];
            }
        }
        for (int i = 0; i < n_caracteristicas; ++i) {
            for (int j = 0; j < n_itens; ++j) {
                R_buffer[i * n_itens + j] = R[i][j];
            }
        }
    }

    MPI_Bcast(&L_buffer[0], n_usuarios * n_caracteristicas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&R_buffer[0], n_itens * n_caracteristicas, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        L.resize(n_usuarios, vector<double>(n_caracteristicas));
        R.resize(n_caracteristicas, vector<double>(n_itens));
        for (int i = 0; i < n_usuarios; ++i) {
            for (int j = 0; j < n_caracteristicas; ++j) {
                L[i][j] = L_buffer[i * n_caracteristicas + j];
            }
        }
        for (int i = 0; i < n_caracteristicas; ++i) {
            for (int j = 0; j < n_itens; ++j) {
                R[i][j] = R_buffer[i * n_itens + j];
            }
        }
    }

    for (int iter = 0; iter < n_iteracoes; ++iter) {
        atualizar_matrizes_gradiente(L, R, entradas, n_caracteristicas, alpha);
    }

    if (rank == 0) {
        double erro_quadratico = calcular_erro_quadratico(L, R, entradas);
        cout << "Erro Quadrático Total: " << erro_quadratico << endl;
    }

    MPI_Finalize();

    return 0;
}
