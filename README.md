# DLProject

1. Create Virtual Environment and Install Dependencies
```
python -m venv .env
source .env/bin/activate  # for linux/macos
.env\Scripts\activate  # for windows
pip install -r requirements.txt
```

## Results

1. Bidirectional LSTM + 0.5 Dropout - 93.71% 
Autoencoder(
  (encoder): Encoder(
    (LSTM1): LSTM(9, 360, batch_first=True, bidirectional=True)
    (dropout): Dropout(p=0.2, inplace=False)
    (LSTM2): LSTM(720, 360, batch_first=True, bidirectional=True)
  )
  (decoder): Decoder(
    (LSTM1): LSTM(360, 360, batch_first=True, bidirectional=True)
    (dropout): Dropout(p=0.2, inplace=False)
    (LSTM2): LSTM(720, 360, batch_first=True, bidirectional=True)
    (output): Linear(in_features=720, out_features=9, bias=True)
  )
  (loss): MSELoss()
)

2. Bidirectional LSTM + 0.2 Dropout - 93.69%