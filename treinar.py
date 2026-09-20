import cv2
import os
import shutil
import random
import torch
from ultralytics import YOLO


PASTA_IMAGENS = "teste"
PASTA_DATASET = "dataset"

EXTENSOES = [".jpg", ".jpeg", ".png"]


def criar_pastas():

    pastas = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val"
    ]

    for pasta in pastas:
        os.makedirs(pasta, exist_ok=True)


def anotar_imagens():

    imagens = []

    for arquivo in os.listdir(PASTA_IMAGENS):

        extensao = os.path.splitext(arquivo)[1].lower()

        if extensao in EXTENSOES:
            imagens.append(arquivo)

    imagens.sort()

    anotacoes = []

    print()
    print("======================================")
    print("          ANOTAÇÃO DOS GATOS")
    print("======================================")
    print()
    print("Desenhe uma caixa ao redor do gato.")
    print("Pressione ENTER para confirmar.")
    print("Pressione ESC para pular.")
    print("Pressione Q para sair.")
    print()

    for i, nome_imagem in enumerate(imagens):

        caminho = os.path.join(
            PASTA_IMAGENS,
            nome_imagem
        )

        imagem = cv2.imread(caminho)

        if imagem is None:
            continue

        print(
            f"[{i + 1}/{len(imagens)}] {nome_imagem}"
        )

        x, y, w, h = cv2.selectROI(
            "Marque o gato",
            imagem,
            showCrosshair=True,
            fromCenter=False
        )

        cv2.destroyAllWindows()

        if w == 0 or h == 0:
            print("Imagem ignorada.")
            continue

        altura, largura = imagem.shape[:2]

        centro_x = (x + w / 2) / largura
        centro_y = (y + h / 2) / altura

        largura_normalizada = w / largura
        altura_normalizada = h / altura

        anotacoes.append({
            "imagem": nome_imagem,
            "x": centro_x,
            "y": centro_y,
            "w": largura_normalizada,
            "h": altura_normalizada
        })

        print("Gato anotado!")

    cv2.destroyAllWindows()

    return anotacoes


def criar_dataset(anotacoes):

    random.shuffle(anotacoes)

    quantidade_treino = int(
        len(anotacoes) * 0.8
    )

    treino = anotacoes[:quantidade_treino]
    validacao = anotacoes[quantidade_treino:]

    print()
    print("======================================")
    print("          DATASET")
    print("======================================")
    print()

    print(f"Total: {len(anotacoes)}")
    print(f"Treino: {len(treino)}")
    print(f"Validação: {len(validacao)}")

    def copiar(lista, tipo):

        for anotacao in lista:

            imagem = anotacao["imagem"]

            nome = os.path.splitext(imagem)[0]

            origem = os.path.join(
                PASTA_IMAGENS,
                imagem
            )

            destino = os.path.join(
                PASTA_DATASET,
                "images",
                tipo,
                imagem
            )

            shutil.copy2(
                origem,
                destino
            )

            label = os.path.join(
                PASTA_DATASET,
                "labels",
                tipo,
                nome + ".txt"
            )

            with open(label, "w") as arquivo:

                arquivo.write(
                    f"0 "
                    f"{anotacao['x']:.6f} "
                    f"{anotacao['y']:.6f} "
                    f"{anotacao['w']:.6f} "
                    f"{anotacao['h']:.6f}\n"
                )

    copiar(treino, "train")
    copiar(validacao, "val")


def criar_yaml():

    with open("dataset.yaml", "w") as arquivo:

        arquivo.write(
            """path: dataset
train: images/train
val: images/val

names:
  0: gato
"""
        )


def treinar():

    print()
    print("======================================")
    print("          TREINAMENTO")
    print("======================================")
    print()

    if torch.cuda.is_available():

        device = 0
        print("GPU NVIDIA detectada.")

    else:

        device = "cpu"
        print("Usando CPU.")

    model = YOLO("yolov8m.pt")

    model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=2,
        device=device,
        name="campusguard_gato",
        pretrained=True,
        patience=10
    )

    print()
    print("======================================")
    print("       TREINAMENTO CONCLUÍDO")
    print("======================================")
    print()

    print(
        "Modelo:"
    )

    print(
        "runs/detect/campusguard_gato/weights/best.pt"
    )


def main():

    if not os.path.exists(PASTA_IMAGENS):

        print(
            f"A pasta '{PASTA_IMAGENS}' não existe."
        )

        return

    criar_pastas()

    anotacoes = anotar_imagens()

    if len(anotacoes) < 2:

        print("Poucas imagens anotadas.")
        return

    criar_dataset(anotacoes)

    criar_yaml()

    treinar()


if __name__ == "__main__":
    main()