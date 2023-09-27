from keras.models import load_model

from A1.A1_new import A1_training
from A2.A2_new import A2_training
from B1.B1 import B1_training
from B2.B2 import B2_training
 

def main():
    
    A1_training()
    model = load_model('model_A1.keras')
    model.summary()
    A2_training()
    model = load_model('model_A2.keras')
    model.summary()
    B1_training()
    model = load_model('model_B1.keras')
    model.summary()
    B2_training()
    model = load_model('model_B2.keras')
    model.summary()

if __name__ == "__main__":
    main()