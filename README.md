# CalcularDistancia

Calcular a Distância entre o Rosto de Duas Pessoas usando Dlib.

## Instalação

Clone o repositório do projeto com o comando:
```bash
git clone https://github.com/joaops/CalcularDistancia.git CalcularDistancia
```

É necessário configurar o ambiente de desenvolvimento seguindo o tutorial:
[Instalação do Dlib com OpenBLAS no Windows para C++](https://joaops.com.br/blog/instalacao-do-dlib-com-openblas-no-windows-para-c)

## Execução

Baixe os seguintes arquivos e extraia eles dentro da pasta build:
[dlib_face_recognition_resnet_model_v1.dat.bz2](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
[shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Então execute o programa clicando duas vezes nele, ou com o comando:
```bash
cd CalcularDistancia
cd build
main.exe
```

Também é possível passar as imagem por parâmetro:
```bash
main.exe pessoas\2-12.jpg pessoas\7-12.jpg
```

## Obeservações

O programa foi compilado para o Windows 10 x64, não irá funcionar na arquitetura x86.
O programa também funciona no Windows 7 x64.

## Links Úteis

[Arquivos do Dlib](http://dlib.net/files/)
[Tutorial de Instalação do Dlib com OpenBLAS no Windows para C++](https://joaops.com.br/blog/instalacao-do-dlib-com-openblas-no-windows-para-c)