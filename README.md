# NEX-ELM

Implementação reprodutível do **NEX-ELM**, uma técnica de explicabilidade glocal e específica para ensembles de **Extreme Learning Machines (ELMs)**. O algoritmo combina explicações locais baseadas em trajetórias intervencionais com uma biblioteca finita de protótipos explicativos condicionados à classe prevista.

O arquivo principal deste projeto é:

```text
nexlm_v68_journal_complete_datasets.py
```

A execução sem argumentos reproduz o protocolo completo usado no artigo: **9 configurações de datasets × 30 seeds = 270 execuções principais**, com CUDA, auditoria numérica CPU–GPU, análises estatísticas, tabelas consolidadas e relatório em PDF.

---

## 1. Visão geral da técnica

O NEX-ELM foi desenvolvido para reduzir a distância entre dois níveis de explicação:

- **local**: por que o ensemble produziu determinada previsão para uma observação específica;
- **global**: quais padrões explicativos recorrentes caracterizam o comportamento do modelo em uma classe ou no conjunto de dados.

Em vez de produzir apenas um ranking global único, o NEX-ELM preserva explicações específicas por instância e organiza padrões locais recorrentes em uma biblioteca pequena de protótipos.

### 1.1 Componentes principais

1. **Ensemble ELM preditivo**  
   Um conjunto de ELMs é treinado com pesos uniformes. Cada estimador usa uma camada oculta aleatória e resolve os pesos de saída por regularização de Ridge.

2. **IRP-NEX local**  
   Para cada observação, o algoritmo explica a probabilidade da classe prevista pelo ensemble. A classe-alvo permanece fixa durante toda a trajetória de perturbação, mesmo que alguma entrada perturbada atravesse uma fronteira de decisão.

3. **Jogo interventional**  
   A contribuição dos atributos é avaliada sob um background empírico. O algoritmo procura uma ordem completa de inserção dos atributos que recupere de modo fiel a saída do modelo.

4. **Assinatura explicativa registrada**  
   A ordem local é convertida em uma assinatura baseada em prefixos registrados. Essa representação permite comparar duas explicações completas com uma distância definida sobre os conjuntos prefixados.

5. **Biblioteca de protótipos por classe**  
   As assinaturas obtidas apenas no conjunto de calibração são agrupadas por classe prevista. Cada classe recebe até quatro medoids, aprendidos por k-medoids determinístico.

6. **Roteamento glocal**  
   Durante a inferência, a classe prevista seleciona a biblioteca candidata. A assinatura local da nova observação é associada ao protótipo mais próximo da classe correspondente.

O resultado final é composto por:

- uma explicação local específica para a observação;
- um protótipo representativo do padrão explicativo recorrente;
- uma visão global formada pela coleção ponderada de protótipos condicionados à classe.

---

## 2. Separação entre treino, calibração e teste

O protocolo impede que o teste externo influencie a construção das explicações.

1. O conjunto completo é dividido em **80% para treino externo** e **20% para teste externo**.
2. O treino externo é dividido novamente usando `calibration_fraction = 0.30`.
3. Isso corresponde, em termos aproximados do conjunto completo, a:

| Partição | Proporção aproximada | Utilização |
|---|---:|---|
| Ajuste do ensemble | 56% | Treinamento dos ELMs |
| Calibração | 24% | Explicações de calibração, k-medoids, protótipos e parâmetros auxiliares |
| Teste externo | 20% | Avaliação preditiva, fidelidade e tempo |

O teste externo não é usado para:

- treinar o ensemble;
- escolher o número de protótipos;
- aprender os medoids;
- selecionar parâmetros da técnica;
- alterar a distância registrada;
- escolher o protótipo com base no rótulo verdadeiro;
- otimizar fidelity a partir do teste.

---

## 3. Comparadores e métricas

### 3.1 Métodos comparados

| Escopo | Métodos |
|---|---|
| Explicação local | NEX-ELM, Kernel SHAP e Random |
| Explicação global | NEX-ELM, Kernel SHAP agregado e X-ELM |
| Tempo local | NEX-ELM local versus Kernel SHAP local |
| Workflow completo | NEX-ELM completo versus X-ELM + Kernel SHAP |

O projeto não afirma que o NEX-ELM isolado é mais rápido que o X-ELM isolado. A comparação de tempo global é feita no nível do workflow necessário para fornecer explicações locais e globais.

### 3.2 Fidelidade

A fidelidade é avaliada em duas direções:

- **Deletion AUC**: mede a deterioração da saída ao remover primeiro os atributos considerados mais importantes;
- **Insertion AUC**: mede a recuperação da saída ao inserir primeiro os atributos considerados mais importantes;
- **Composite AUC**: síntese registrada das trajetórias de deletion e insertion.

Valores maiores representam melhor fidelidade no protocolo implementado.

### 3.3 Evidência estatística

A suíte inclui:

- teste de Shapiro–Wilk;
- teste t pareado unilateral;
- Wilcoxon signed-rank unilateral;
- teste de permutação pareado;
- ANOVA de medidas repetidas;
- teste de Friedman;
- Kendall’s W;
- pós-teste de Nemenyi;
- teste binomial exato de taxa de vitórias;
- intervalos exatos de Clopper–Pearson;
- correção de Holm para múltiplos testes.

A regra confirmatória considera em conjunto direção, intervalo de confiança, testes pareados, tamanho de efeito e consistência entre seeds.

---

## 4. Datasets executados no plano completo

| Identificador no código | Amostras | Atributos | Classes | Origem/observação |
|---|---:|---:|---:|---|
| `electrical_grid_stability` | 10.000 | 13 | 2 | UCI; mantém `stab` para reprodução do cenário X-ELM |
| `electrical_grid_stability_without_stab` | 10.000 | 12 | 2 | Controle de proxy; remove `stab` |
| `pima_indians_diabetes` | 768 | 8 | 2 | Pima Indians Diabetes |
| `wisconsin_breast_cancer_original` | 683 | 9 | 2 | Wisconsin Breast Cancer Original após limpeza |
| `ionosphere_binary` | 351 | 34 | 2 | UCI Ionosphere |
| `wine_multiclass` | 178 | 13 | 3 | Dataset Wine do scikit-learn |
| `iris_multiclass` | 150 | 4 | 3 | Dataset Iris do scikit-learn |
| `digits_multiclass` | 1.797 | 64 | 10 | Dataset Digits do scikit-learn |
| `breast_cancer_diagnostic` | 569 | 30 | 2 | Breast Cancer Diagnostic do scikit-learn |

Os datasets do scikit-learn são carregados localmente pela biblioteca. Pima, Wisconsin Original, Electrical Grid e Ionosphere podem ser baixados pelo próprio script para o diretório `reference_data`.

---

## 5. Configuração usada para gerar os resultados do artigo

A tabela abaixo apresenta os valores efetivos resolvidos pelo arquivo anexado ao executar o plano completo sem argumentos.

### 5.1 Protocolo experimental e modelo

| Componente | Configuração efetiva |
|---|---|
| Versão | `v68` |
| Método | `GloPro-Complete` |
| Plano de estudo | `complete` |
| Modo | `real` |
| Datasets | 9 configurações |
| Repetições por dataset | 30 |
| Total de execuções principais | 270 |
| Seed inicial | 100000 |
| Passo entre seeds | 1009 |
| Última seed | 129261 |
| Divisão externa de teste | 20% |
| Fração de calibração dentro do treino externo | 30% |
| Número de estimadores ELM | 400 |
| Neurônios ocultos por ELM | 50 |
| Ativação | `tanh` |
| Regularização Ridge | `1e-2` |
| Peso dos estimadores | uniforme |
| Número máximo de protótipos por classe | 4 |
| Mínimo de linhas de calibração por slot de protótipo | 2 |
| Agrupamento | k-medoids determinístico |
| Organização da biblioteca | condicionada à classe prevista |
| Taxa mínima registrada de vitórias | 0,80 |
| Vitórias mínimas em 30 seeds | 24 |
| Nível de significância | 0,05 |
| Tamanho de efeito mínimo registrado | 0,50 |

### 5.2 Explicação e avaliação

| Parâmetro | Valor efetivo |
|---|---:|
| Protocolo de ablação | `expected_background` |
| Background do Kernel SHAP | 350 |
| Instâncias locais explicadas pelo Kernel SHAP | 50 |
| `shap_nsamples` | 0, delegado ao SHAP |
| Nós probabilísticos NEX | 16 |
| Nós da análise de convergência | `8,16,32` |
| Instâncias da análise de convergência | 2 |
| Explicações globais de calibração | 96 |
| Bootstrap global | 128 |
| Instâncias de tuning | 30 |
| Candidatos de interação | 5 |
| Grid de interação | `0,0.25,0.50,0.75` |
| Grid do candidate pool | `50` |
| Grid de redundância | `0,0.05,0.10,0.20` |
| Peso de deletion na calibração | 0,50 |
| Instâncias globais de teste | 100 |
| Repetições do controle aleatório de fidelidade | 5 |
| Iterações de bootstrap inferencial | 5.000 |
| Iterações do teste de permutação | 100.000 |
| Margem de não inferioridade | 0,01 |

### 5.3 CUDA e execução

| Componente | Configuração efetiva |
|---|---|
| Dispositivo padrão | `cuda` |
| Perfil de GPU | `rtx3060_12gb` |
| Precisão | `float32` |
| TF32 | habilitado |
| Alocador CUDA | `cudaMallocAsync` |
| Batch de predição GPU | 32.768 |
| Batch de estimadores | 256 |
| Batch de instâncias NEX | 4 |
| Bloco de background NEX | 256 |
| Batch de gradientes NEX | 16.384 |
| Batch de tarefas de ablação | 1.024 |
| Batch mínimo após OOM | 1.024 |
| Linhas de warm-up | 512 |
| Warm-up CUDA | habilitado |
| `torch.compile` | desabilitado |
| Auditoria CPU–CUDA | habilitada |
| Tolerância da auditoria | `2e-4` |

### 5.4 Ambiente experimental informado

| Recurso | Ambiente usado nos experimentos |
|---|---|
| Sistema operacional | Windows 10 |
| Processador | Intel Core i9 de 10ª geração |
| Memória RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 3060 com 12 GB |
| PyTorch | 2.12.1 + CUDA 12.6 |

> O código não fixa a versão do Python nem todas as versões das dependências em um arquivo de lock. Para reprodução estrita, registre as versões do ambiente usado por meio de `pip freeze > requirements-lock.txt`.

---

## 6. Requisitos

### 6.1 Requisitos mínimos de software

- Python 3.10 ou 3.11 recomendado;
- NumPy;
- pandas;
- SciPy;
- Matplotlib;
- scikit-learn;
- SHAP;
- PyTorch;
- ReportLab, para gerar `relatorio.pdf`.

### 6.2 Instalação em ambiente virtual

#### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

#### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Instale primeiro uma versão do PyTorch compatível com sua GPU e sua instalação CUDA. Em seguida, instale as demais dependências:

```bash
pip install numpy pandas scipy matplotlib scikit-learn shap reportlab
```

Para uma execução apenas em CPU:

```bash
pip install torch numpy pandas scipy matplotlib scikit-learn shap reportlab
```

Verifique o ambiente:

```bash
python -c "import torch, shap, sklearn, numpy, pandas, scipy; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
```

---

## 7. Organização recomendada do projeto

```text
projeto_nex_elm/
├── nexlm_v68_journal_complete_datasets.py
├── README.md
├── reference_data/
└── resultados/
```

O núcleo matemático está incorporado no arquivo principal. Um arquivo adjacente chamado `nexlm_v68_glopro_complete_core.py` é opcional e só é carregado quando seu SHA-256 coincide com o hash esperado:

```text
fc78619795db892180c7b978366a67b15a48741103f0fcc98eeab7b7dcf2031f
```

Caso um núcleo adjacente tenha hash diferente, ele será ignorado e o núcleo incorporado, verificado por integridade, será usado.

---

## 8. Como executar

### 8.1 Exibir a ajuda

```bash
python nexlm_v68_journal_complete_datasets.py --help
```

### 8.2 Reproduzir o experimento completo

A execução sem argumentos usa CUDA, o perfil RTX 3060 de 12 GB, 30 repetições e todos os nove cenários:

```bash
python nexlm_v68_journal_complete_datasets.py
```

Para definir o diretório de saída:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --output-dir resultados_nex_elm_v68
```

No Windows PowerShell, o mesmo comando pode ser escrito em uma linha:

```powershell
python nexlm_v68_journal_complete_datasets.py --output-dir resultados_nex_elm_v68
```

> O plano `complete` exige que o dispositivo resolvido seja CUDA. O script interrompe a execução se o plano completo for solicitado sem GPU CUDA disponível.

### 8.3 Teste rápido com GPU

O parâmetro `--quick` reduz cada fase a uma repetição e é indicado para testar dependências, downloads e criação de arquivos:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --quick \
  --output-dir resultados_smoke_test
```

### 8.4 Teste rápido em CPU

Para CPU, use um plano não padrão, como `custom`:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --study-plan custom \
  --datasets iris \
  --n-repeats 1 \
  --device cpu \
  --skip-pdf-report \
  --output-dir resultados_cpu_teste
```

O nome `iris` é convertido para `iris_multiclass` pelo sistema de aliases.

### 8.5 Executar datasets específicos

Exemplo com três datasets e cinco seeds:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --study-plan custom \
  --datasets iris,wine,breast_diagnostic \
  --n-repeats 5 \
  --device cuda \
  --output-dir resultados_customizados
```

Aliases aceitos:

| Alias | Dataset canônico |
|---|---|
| `grid` | `electrical_grid_stability` |
| `grid_without_stab` | `electrical_grid_stability_without_stab` |
| `pima` | `pima_indians_diabetes` |
| `wisconsin` | `wisconsin_breast_cancer_original` |
| `ionosphere` | `ionosphere_binary` |
| `wine` | `wine_multiclass` |
| `iris` | `iris_multiclass` |
| `digits` | `digits_multiclass` |
| `breast_diagnostic` | `breast_cancer_diagnostic` |

### 8.6 Executar apenas a replicação

```bash
python nexlm_v68_journal_complete_datasets.py \
  --study-plan replication \
  --replication-repeats 30 \
  --replication-random-state 100000 \
  --output-dir resultados_replicacao
```

Essa fase usa, por padrão:

- Electrical Grid com `stab`;
- Pima Indians Diabetes;
- Wisconsin Breast Cancer Original.

### 8.7 Executar apenas a generalização

```bash
python nexlm_v68_journal_complete_datasets.py \
  --study-plan generalization \
  --generalization-repeats 30 \
  --generalization-random-state 100000 \
  --output-dir resultados_generalizacao
```

### 8.8 Usar arquivos locais sem download

Coloque os arquivos no diretório indicado por `--data-dir` e use:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --data-dir reference_data \
  --no-download
```

Nomes reconhecidos:

```text
reference_data/
├── Data_for_UCI_named.csv
├── pima-indians-diabetes.data.csv
├── breast-cancer-wisconsin.data
└── ionosphere.data
```

### 8.9 Perfil conservador de GPU

Quando o perfil padrão consumir memória demais, tente:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --gpu-profile conservative \
  --output-dir resultados_gpu_conservador
```

O perfil altera apenas parâmetros de execução e batching. A matemática central permanece protegida pelo núcleo verificado.

### 8.10 Gerar novamente apenas as estatísticas

```bash
python nexlm_v68_journal_complete_datasets.py \
  --statistics-only-from resultados_nex_elm_v68 \
  --statistics-output-dir resultados_reanalise
```

O diretório informado deve conter `seed_metrics.csv` e, quando disponível, `estatistica_entre_seeds.csv`, diretamente ou em `combined/tabelas`.

### 8.11 Gerar novamente apenas o relatório PDF

```bash
python nexlm_v68_journal_complete_datasets.py \
  --report-only-from resultados_nex_elm_v68 \
  --report-output relatorio_nex_elm.pdf
```

Para executar o experimento sem gerar o PDF:

```bash
python nexlm_v68_journal_complete_datasets.py --skip-pdf-report
```

---

## 9. Estrutura dos resultados

A estrutura geral de saída é:

```text
resultados_v68_glopro_complete_YYYYMMDD_HHMMSS/
├── plano_confirmatorio.csv
├── plano_validacao_completa_v68.csv
├── registro_seeds_v68.csv
├── registro_protocolo_confirmatorio.json
├── limites_de_alegacao_v68.csv
├── relatorio.pdf
├── per_seed/
│   └── <dataset>/<cenario>/seed_<valor>/
│       ├── manifest.json
│       └── tabelas/
└── combined/
    ├── tabelas/
    ├── graficos_estatisticos/
    ├── graficos_preditivos/
    └── outros artefatos consolidados/
```

### 9.1 Arquivos preditivos principais

| Arquivo | Conteúdo |
|---|---|
| `predictive_performance_per_seed.csv` | Métricas preditivas por seed |
| `predictive_performance_summary.csv` | Médias e intervalos entre seeds |
| `predictive_performance_article_table.csv` | Tabela pronta para o artigo |
| `predictive_confusion_matrix.csv` | Matrizes de confusão completas |
| `predictive_confusion_matrix_summary.csv` | Resumo das matrizes de confusão |
| `predictive_class_metrics.csv` | Métricas por classe |
| `predictive_class_metrics_summary.csv` | Resumo das métricas por classe |
| `predictive_dataset_summary.csv` | Dimensões e características dos datasets |

### 9.2 Explicabilidade e protótipos

| Arquivo | Conteúdo |
|---|---|
| `local_fidelity_summary.csv` | Fidelidade local por método e seed |
| `global_fidelity_summary.csv` | Fidelidade global por método e seed |
| `local_agreement.csv` | Concordância estrutural local |
| `global_importance.csv` | Importância global |
| `global_class_importance.csv` | Importância global por classe |
| `prototype_library.csv` | Biblioteca completa de protótipos |
| `prototype_library_class_summary.csv` | Resumo por classe |
| `prototype_routing.csv` | Roteamento das observações |
| `prototype_routing_summary.csv` | Resumo do roteamento |
| `prototype_stability_between_seeds.csv` | Estabilidade dos protótipos |

### 9.3 Estatística, tempo e auditoria

| Arquivo | Conteúdo |
|---|---|
| `seed_metrics.csv` | Tabela central para inferência entre seeds |
| `inferential_test_summary.csv` | Síntese dos testes confirmatórios |
| `paired_t_tests.csv` | Testes t pareados |
| `shapiro_wilk_tests.csv` | Diagnóstico de normalidade |
| `repeated_measures_anova.csv` | ANOVA de medidas repetidas |
| `friedman_tests.csv` | Testes de Friedman |
| `average_method_ranks.csv` | Ranks médios |
| `nemenyi_posthoc.csv` | Pós-teste de Nemenyi |
| `binomial_win_rate_tests.csv` | Taxas de vitória e Clopper–Pearson |
| `multiple_testing_families.csv` | Famílias de correção múltipla |
| `timing.csv` | Tempos locais e de workflow |
| `cuda_audit.csv` | Equivalência numérica CPU–GPU |
| `gpu_runtime.csv` | Informações de execução e memória GPU |

---

## 10. Integridade e reprodutibilidade

Antes de executar o experimento, o script:

1. carrega o núcleo matemático incorporado ou um núcleo adjacente com hash idêntico;
2. verifica o SHA-256;
3. executa self-tests;
4. congela a configuração de protótipos em `K = 4`;
5. registra o protocolo e a configuração pública;
6. impede sobreposição não autorizada com a bateria confirmatória anterior;
7. registra seeds, cenários e limites de alegação.

Não é permitido alterar o número fixo de protótipos pela CLI. Os comandos abaixo geram erro:

```bash
python nexlm_v68_journal_complete_datasets.py --prototype-count 3
python nexlm_v68_journal_complete_datasets.py --prototype-min-calibration-rows-per-slot 4
```

Os únicos valores aceitos nesta versão são:

```text
prototype-count = 4
prototype-min-calibration-rows-per-slot = 2
```

---

## 11. Solução de problemas

### `The shap package is required`

```bash
pip install shap
```

### `The torch package is required for NEX-ELM`

Instale o PyTorch compatível com CPU ou CUDA antes de iniciar o experimento.

### `The default complete experiment requires CUDA`

O plano completo não executa em CPU. Para uma auditoria em CPU, use:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --study-plan custom \
  --datasets iris \
  --n-repeats 1 \
  --device cpu
```

### Erro de falta de memória CUDA

Tente o perfil conservador:

```bash
python nexlm_v68_journal_complete_datasets.py --gpu-profile conservative
```

Também é possível usar `--gpu-profile custom` e reduzir os batches, por exemplo:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --gpu-profile custom \
  --gpu-batch-size 8192 \
  --estimator-batch-size 32 \
  --nex-instance-batch-size 1 \
  --nex-background-block-size 64
```

### Arquivo de dataset não encontrado com `--no-download`

Confirme o caminho passado em `--data-dir` e os nomes dos arquivos listados na Seção 8.8.

### O relatório PDF não foi gerado

Instale o ReportLab:

```bash
pip install reportlab
```

Ou execute com:

```bash
python nexlm_v68_journal_complete_datasets.py --skip-pdf-report
```

### Resultados diferentes entre máquinas

Pequenas diferenças podem ocorrer por versão de biblioteca, CUDA, TF32, hardware e paralelismo. Para documentar o ambiente:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pip freeze > requirements-lock.txt
```

---

## 12. Limites de interpretação

Os resultados sustentam alegações sobre:

- fidelidade de explicação sob o jogo interventional registrado;
- granularidade local, global e glocal;
- estabilidade entre seeds;
- custo computacional no ambiente auditado.

Eles não demonstram, por si só:

- causalidade;
- justiça algorítmica;
- utilidade para usuários humanos;
- superioridade em toda técnica de XAI;
- superioridade de tempo sobre o X-ELM isolado;
- ausência universal de memory leak;
- generalização automática para imagens, texto, séries temporais ou modelos não ELM.

---

## 13. Citação

Ao utilizar o código em trabalhos acadêmicos, cite o artigo associado ao NEX-ELM. Após a publicação, substitua este bloco pelos dados bibliográficos e DOI definitivos.

```bibtex
@article{nexelm,
  title   = {NEX-ELM: Advancing Glocal Explainable AI for Extreme Learning Machines},
  author  = {Author information omitted for double-blind review},
  journal = {Neural Computing and Applications},
  year    = {forthcoming}
}
```

---

## 14. Licença

Nenhum arquivo de licença foi fornecido com o código anexado. Antes de distribuir o projeto publicamente, inclua uma licença compatível com a política dos autores, das dependências e dos datasets utilizados.
