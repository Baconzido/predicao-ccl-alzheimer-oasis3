# data/

Esta pasta é **intencionalmente vazia** de dados de imagem. O dataset
[OASIS-3](https://www.oasis-brains.org/) é distribuído sob um acordo de uso
(Data Use Agreement) que **proíbe redistribuição** — por isso nenhum arquivo
de imagem do OASIS-3 é versionado neste repositório.

## Como solicitar acesso ao OASIS-3

1. Acesse [oasis-brains.org](https://www.oasis-brains.org/) e crie uma conta.
2. Solicite acesso ao **OASIS-3** e aceite o Data Use Agreement.
3. Após aprovação, crie uma conta no [NITRC](https://www.nitrc.org/) e/ou no
   [XNAT Central](https://central.xnat.org) (mesmo e-mail usado no OASIS-3)
   para ter acesso de download via NITRC/XNAT.

## Como organizar os arquivos localmente

A lista de sujeitos usada neste projeto está em
[`download_list.txt`](download_list.txt) (61 sujeitos: 31 conversores/pMCI,
30 estáveis/sMCI). Depois de baixar os dados (ver
`[preencher: link para o script de download, mantido no repositório GitHub
TCC desta conta]`), organize localmente assim (essas pastas **não** devem
ser commitadas — já estão no `.gitignore`):

```
data/
├── README.md              (este arquivo, versionado)
├── download_list.txt       (versionado - só a lista de IDs, sem dados)
├── oasis3_mri_data/        (NÃO versionado - .zip extraídos/NIfTI brutos)
└── oasis3_preprocessed/    (NÃO versionado - volumes .npy pré-processados)
```

Depois de organizar os dados, siga o Pipeline descrito no
[README principal](../README.md) para extrair, pré-processar e treinar.
