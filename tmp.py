
import dill as pk
data = pk.load(open("/home/data/ssh5131/code/naver/EM_SAIS_klueroberta/dataset/for_test_noET/Processed/dev_inputs.pkl", 'rb'))

print(len(data))