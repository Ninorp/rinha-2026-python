# Rinha de Backend 2026 - Python

Implementacao em Python para a Rinha de Backend 2026, usando uma API pequena em
Robyn, serializacao rapida com `msgspec`, calculo vetorial com NumPy e
gerenciamento de dependencias com `uv`.

O projeto esta estruturado para manter o Python como camada de orquestracao da
API e deixar o trabalho pesado concentrado em arrays compactos. A primeira
versao usa kNN exato sobre vetores quantizados em `uint8`; essa base ja reduz o
uso de memoria e deixa um caminho claro para trocar o miolo de busca por ANN/IVF
sem mudar o contrato HTTP.

## Caracteristicas

- API HTTP com [Robyn](https://robyn.tech/).
- Parsing e serializacao JSON com `msgspec`.
- Vetorizacao da transacao para 14 dimensoes.
- Indice de referencias pre-processado em arquivos NumPy.
- Vetores quantizados de `float32` para `uint8` para reduzir memoria.
- Busca kNN exata em blocos, evitando alocar uma matriz gigante por request.
- `docker-compose.yml` com Nginx + duas instancias da API.
- Dependencias e lockfile gerenciados por `uv`.
- Testes unitarios e de handlers da API com `pytest`.
- CI no GitHub Actions.

## Arquitetura

```text
cliente
  |
  v
nginx :9999
  |
  +-- api1 :8080  Robyn + msgspec + NumPy
  |
  +-- api2 :8080  Robyn + msgspec + NumPy
```

Cada instancia da API carrega o indice de referencias no startup. Se o indice
ainda nao existir, a aplicacao tenta cria-lo a partir de
`resources/references.json.gz`. Na ausencia do dataset oficial, ela usa
`resources/example-references.json`, que existe apenas para desenvolvimento e
testes locais.

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
quantiza o vetor da consulta
  |
  v
busca os 5 vizinhos mais proximos no indice
  |
  v
calcula fraud_score
  |
  v
retorna approved true/false
```

O limiar atual e simples:

```text
fraud_score < 0.6  => approved = true
fraud_score >= 0.6 => approved = false
```

## Estrutura

```text
.
+-- .github/workflows/ci.yml      # pipeline de testes e validacao do compose
+-- Dockerfile                    # imagem da API com uv
+-- docker-compose.yml            # nginx + duas APIs
+-- nginx.conf                    # balanceamento simples para as APIs
+-- pyproject.toml                # metadata, dependencias e config do pytest
+-- uv.lock                       # lockfile do uv
+-- resources/
|   +-- example-references.json   # dataset minimo para desenvolvimento
|   +-- mcc_risk.json             # risco default por MCC
|   +-- normalization.json        # parametros de normalizacao
+-- scripts/
|   +-- build_index.py            # gera arquivos de indice em resources/index
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
resources/index/labels.u1.npy
```

`resources/index/` e ignorado pelo Git porque e artefato gerado.

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
| `ROBYN_HOST` | definido no compose | Host do servidor Robyn |
| `ROBYN_PORT` | definido no compose | Porta do servidor Robyn |

## Pontos de Evolucao

O principal gargalo futuro e a busca vetorial. A implementacao atual e uma base
compacta e previsivel, mas ainda faz kNN exato em blocos. Para perseguir p99
abaixo de 30ms com o dataset completo, os proximos passos naturais sao:

- criar um indice ANN/IVF com centroides;
- buscar apenas nas celulas mais proximas;
- fazer rerank exato em poucos milhares de candidatos;
- medir recall versus latencia usando uma amostra do dataset oficial;
- ajustar quantizacao, threshold e numero de candidatos.

Arquivos mais importantes para essa evolucao:

- `src/rinha_api/index.py`
- `scripts/build_index.py`
- `resources/normalization.json`
- `resources/mcc_risk.json`
