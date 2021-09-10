# ML_optimization
機械学習で使われる最適化アルゴリズムの勉強をする．

## ディレクトリ構成
```
|- .devcontainer/ -> dockerの設定ファイル類
|- note/
    |- lib/ -> note/の階層のipynbで使うクラス等を定義したファイル類
```

## 実行環境
- [nvidia/cuda:11.0.3-devel-ubuntu20.04](https://hub.docker.com/layers/nvidia/cuda/11.0-devel-ubuntu20.04/images/sha256-bd7a97c99c7a2bcc183ea07e6b193727de3d180b8c8d118575c6a7968d30c80c?context=explore) に python3 と pytorch 等を入れて使っている．
    - 詳しくは `dockerfile` を参照．

## Memo
- `docker-compose`のバージョンを上げた (1.29)．
    - `docker-compose.yml`を実行するため．