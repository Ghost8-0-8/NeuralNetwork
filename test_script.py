import subprocess
import sys
import json

def run_test():
    import os
    dossier = "./TESTS"
    print("\nRésultats des tests :")
    print("--------------------")

    for fichier in os.listdir(dossier):
        chemin_complet = os.path.join(dossier, fichier)
        if os.path.isfile(chemin_complet):
            success = 0
            with open("result.json", "r") as file:
                data = json.load(file)
                print(f"\033[36m\033[1m\033[4m{dossier}/{fichier}\033[0m")
                result = subprocess.run(["./my_torch_analyzer", "--predict", "my_torch_network_best2.nn", f"{dossier}/{fichier}"], 
                                    capture_output=True, text=True)
                predictions = result.stdout.strip().split('\n')
                expected_results = data[fichier]
                for number , expected in enumerate(expected_results):
                    actual = predictions[number]
                    status = "\033[32m✓\033[0m" if actual == expected else "\033[31m✗\033[0m"
                    if actual == expected:
                        success += 1
                    print(f"{status} Position n°{number}: Attendu={expected}, Obtenu={actual}")
                total = number + 1
                color = ""
                if 0 <= success/total*100 <= 33: 
                    color = "\033[31m"
                elif 33 < success/total*100 <= 66:
                    color = "\033[33m"
                elif 66 < success/total*100:
                    color = "\033[32m"
                print(f"\n\033[34m\033[4mRésumé\033[0m: {color}{success}/{total}\033[0m tests réussis ({color}{(success/total)*100:.1f}%\033[0m)")












    

if __name__ == "__main__":
    run_test()