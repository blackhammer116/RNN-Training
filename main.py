import os
import sys

def main():
    model = (sys.argv[1] if len(sys.argv) > 1 else "VRNN").lower()
    
    if model == "gru":
        print("GRU model selected.")
        os.system("bash timeseries-weather-data-prep.sh")
        os.system("python GRU-model.py")
    elif model == "vrnn":
        print("VRNN model selected.")
        os.system("python names-prep.py")
        os.system("python VRNN.py")
    elif model == "lstm":
        print("LSTM model selected.")
        os.system("bash imdb-data-prep.sh")
        os.system("python LSTM-model.py")
    else:
        print(f"Model {model} not recognized.")
        return

if __name__ == "__main__":
    main()