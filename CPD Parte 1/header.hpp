#include <vector>

using namespace std;
// Estrutura para armazenar uma entrada do arquivo
struct Entrada ;

// Função para ler as entradas do arquivo
void ler_entrada_arquivo(const char* nome_arquivo, int& n_iteracoes, double& alfa, int& n_caracteristicas, vector<Entrada>& entradas, int& n_usuarios, int& n_itens);

// Função para preencher as matrizes L e R com valores aleatórios
void preenche_aleatorio_LR(int nU, int nI, int nF, vector<vector<double>>& L, vector<vector<double>>& R);

// Função para atualizar as matrizes L e R com base nas entradas e no gradiente
void ActualizarMatriz(vector<vector<double>>& L, vector<vector<double>>& R, const vector<Entrada>& entradas, int n_caracteristicas, double alfa);

// Função para recomendar itens para cada usuário
vector<int> FuncaoRecomendarItems(const vector<vector<double>>& L, const vector<vector<double>>& R);


