# Rinha de Backend 2026 - Python

Implementacao em Python para a Rinha de Backend 2026, usando uma API pequena em
Robyn, serializacao rapida com `msgspec`, calculo vetorial com NumPy e
gerenciamento de dependencias com `uv`.

O projeto esta estruturado para manter o Python como camada de orquestracao da
API e deixar o trabalho pesado concentrado em arrays compactos. A busca usa um
indice IVF-flat: os vetores sao agrupados por centroides no build e, em runtime,
a API consulta um conjunto pequeno de celulas proximas antes de fazer rerank dos
candidatos com vetores `float16`.

## Caracteristicas

- API HTTP com [Robyn](https://robyn.tech/).
- Parsing e serializacao JSON com `msgspec`.
- Vetorizacao da transacao para 14 dimensoes.
- Indice de referencias pre-processado em arquivos NumPy.
- Vetores de rerank em `float16`, com versao `uint8` mantida como artefato
  compacto auxiliar.
- Busca ANN/IVF com `nprobe` ajustado para equilibrar recall e latencia.
- `docker-compose.yml` com HAProxy em modo TCP + duas instancias da API.
- Dependencias e lockfile gerenciados por `uv`.
- Testes unitarios e de handlers da API com `pytest`.
- CI no GitHub Actions.

## Arquitetura

```text
cliente
  |
  v
HAProxy :9999
  |
  +-- api1 :8080  Robyn + msgspec + NumPy
  |
  +-- api2 :8080  Robyn + msgspec + NumPy
```

Cada instancia da API carrega o indice de referencias no startup. No compose de
benchmark, os arrays principais sao pre-carregados em RAM para evitar cauda de
page fault; em desenvolvimento, isso pode ser desligado com
`RINHA_INDEX_PRELOAD=0`. A imagem Docker gera o indice durante o build; se os
arquivos de indice nao existirem, a aplicacao sobe com um indice vazio apenas
para manter o contrato de saude.

## Fluxo da Requisicao

```text
POST /fraud-score
  |
  v
parse JSON com msgspec
  |
  v
extrai features e monta vetor de 14 dimensoes
  |
  v
calcula as celulas IVF mais proximas
  |
  v
arvore confiante tenta responder em O(1)
  |
  +-- incerto: busca 5 vizinhos nas celulas IVF selecionadas
  |
  v
calcula fraud_score
  |
  v
retorna approved true/false
```

O limiar segue a regra oficial de deteccao:

```text
fraud_score < 0.6  => approved = true
fraud_score >= 0.6 => approved = false
```

## Estrutura

```text
.
+-- .github/workflows/ci.yml      # pipeline de testes e validacao do compose
+-- Dockerfile                    # imagem da API com uv
+-- docker-compose.yml            # HAProxy + duas APIs
+-- haproxy.cfg                   # balanceamento TCP round-robin simples
+-- pyproject.toml                # metadata, dependencias e config do pytest
+-- uv.lock                       # lockfile do uv
+-- resources/
|   +-- example-references.json   # dataset minimo para desenvolvimento
|   +-- mcc_risk.json             # risco default por MCC
|   +-- normalization.json        # parametros de normalizacao
+-- scripts/
|   +-- build_index.py            # gera arquivos de indice em resources/index
|   +-- evaluate_quality.py       # mede FP/FN/E localmente contra test-data.json
+-- src/rinha_api/
|   +-- app.py                    # rotas Robyn e inicializacao da aplicacao
|   +-- config.py                 # paths e carga de configuracoes
|   +-- index.py                  # build/load do indice e busca kNN
|   +-- vectorize.py              # transformacao payload -> vetor de 14 dims
+-- tests/
    +-- test_api.py               # testes dos handlers HTTP
    +-- test_vectorize.py         # teste da vetorizacao
```

## Requisitos

- Python 3.12+
- `uv`
- Docker e Docker Compose, para rodar a stack completa

Instalacao local:

```bash
uv sync --frozen
```

## Rodando Localmente

Para rodar a API diretamente:

```bash
uv run python -m rinha_api.app
```

Endpoint de saude:

```bash
curl http://localhost:8080/ready
```

Para rodar a stack da Rinha:

```bash
docker compose up --build
```

A porta publica fica em:

```text
http://localhost:9999
```

## Dataset e Indice

Durante o desenvolvimento, o projeto usa:

```text
resources/example-references.json
```

Para usar o dataset oficial, coloque o arquivo em:

```text
resources/references.json.gz
```

Depois gere o indice:

```bash
uv run python scripts/build_index.py \
  --references resources/references.json.gz \
  --out resources/index
```

Arquivos gerados:

```text
resources/index/vectors.u1.npy
resources/index/vectors.f16.npy
resources/index/labels.u1.npy
resources/index/centroids.f32.npy
resources/index/offsets.i8.npy
resources/index/tree.npz
resources/index/index.json
```

`resources/index/` e ignorado pelo Git porque e artefato gerado.

## Resultado de Referencia

O pacote atual esta no estagio de tuning ANN/IVF com indice pre-carregado,
arvore confiante e normas de vetores/centroides pre-computadas. A configuracao
local de benchmark esta em:

```text
RINHA_IVF_CELLS=8192
RINHA_IVF_NPROBE=10
RINHA_IVF_SAMPLE=250000
RINHA_IVF_ITERATIONS=6
RINHA_TREE_SAMPLE=2000000
RINHA_TREE_DEPTH=14
RINHA_TREE_QUANTILES=511
RINHA_TREE_MIN_LEAF=50
RINHA_TREE_CONFIDENCE=0.90
RINHA_INDEX_PRELOAD=1
RINHA_INDEX_WARMUP=0
```

Melhor rodada local reportada durante o tuning:

```text
p99: 1.60ms
false_positive_detections: 4
false_negative_detections: 4
weighted_errors_E: 16
final_score: 5426.72
```

Esse numero veio do script local de validacao usado no tuning e deve ser
revalidado contra a rodada oficial completa antes de publicar nova imagem. O
resultado publico anterior da previa ainda media uma imagem antiga, com p99 em
torno de 10ms; para a previa refletir este pacote, a imagem
`rodolfoc2s/rinha-2026-python:latest` precisa ser reconstruida e enviada.

O threshold de aprovacao permanece fixo em `fraud_score < 0.6`.

## Testes

```bash
uv run pytest -q
```

Os testes cobrem:

- vetorizacao do payload para 14 dimensoes;
- resposta de `/ready`;
- aprovacao quando `fraud_score` fica abaixo do threshold;
- negacao quando `fraud_score` atinge o threshold.

## CI

O workflow em `.github/workflows/ci.yml` roda em push para `main` e em pull
requests.

Etapas:

```bash
uv sync --frozen
uv run pytest -q
docker compose config
```

## Variaveis de Ambiente

| Variavel | Default | Uso |
| --- | --- | --- |
| `RINHA_RESOURCES_DIR` | `resources` | Diretorio com configs e dataset |
| `RINHA_INDEX_DIR` | `resources/index` | Diretorio dos arquivos de indice |
| `RINHA_IVF_CELLS` | `4096` (`8192` no Dockerfile/compose) | Numero de centroides/celulas gerados no build |
| `RINHA_IVF_NPROBE` | `1` (`10` no compose) | Numero de celulas consultadas por request quando ha fallback para rerank |
| `RINHA_IVF_SAMPLE` | `50000` (`250000` no Dockerfile/compose) | Amostra usada para treinar os centroides |
| `RINHA_IVF_ITERATIONS` | `4` (`6` no Dockerfile/compose) | Iteracoes de k-means sobre a amostra |
| `RINHA_INDEX_PRELOAD` | `0` (`1` no compose) | Carrega arrays do indice em RAM em vez de usar mmap |
| `RINHA_INDEX_WARMUP` | `0` | Faz uma varredura completa do indice no startup antes de `/ready` |
| `RINHA_CELL_FAST_MARGIN` | `1.0` | Atalho antigo por maioria da celula; `1.0` deixa desativado |
| `RINHA_TREE_CONFIDENCE` | `0.95` (`0.90` no compose) | Confianca minima para a arvore responder sem fallback IVF |
| `RINHA_TREE_SAMPLE` | `500000` (`2000000` no Dockerfile/compose) | Amostra usada para treinar a arvore confiante no build |
| `RINHA_TREE_DEPTH` | `10` (`14` no Dockerfile/compose) | Profundidade maxima da arvore confiante |
| `RINHA_TREE_QUANTILES` | `199` (`511` no Dockerfile/compose) | Quantis candidatos por feature no treino da arvore |
| `RINHA_TREE_MIN_LEAF` | `50` | Tamanho minimo de folha no treino da arvore |
| `ROBYN_HOST` | definido no compose | Host do servidor Robyn |
| `ROBYN_PORT` | definido no compose | Porta do servidor Robyn |
| `ROBYN_LOG_LEVEL` | `WARN` | Reduz logs no benchmark |
| `ROBYN_PROCESSES` | `1` | Processos por instancia da API |
| `ROBYN_WORKERS` | `4` (`1` no compose) | Workers por processo da API |

## Pontos de Evolucao

O principal gargalo era a busca vetorial exata sobre o dataset completo. A
implementacao atual aplica ANN/IVF, pre-carregamento opcional e distancia
otimizada por normas pre-computadas. O p99 local ja caiu para a faixa de
1-2ms, mas ainda ha uma cauda de fallback IVF em torno de alguns pontos
percentuais.

Proximos passos naturais:

- rodar a bateria oficial completa com `submission.docker-compose.yml`;
- publicar a imagem atualizada e comparar com a previa publica;
- varrer `RINHA_IVF_NPROBE=10/12/14` para achar o melhor compromisso p99/E;
- investigar um classificador O(1) melhor para os casos de borda que ainda
  caem no IVF;
- reduzir tempo de build do indice, que hoje ainda fica na ordem de minutos
  com `TREE_SAMPLE=2000000`.

Arquivos mais importantes para essa evolucao:

- `src/rinha_api/index.py`
- `scripts/build_index.py`
- `resources/normalization.json`
- `resources/mcc_risk.json`
