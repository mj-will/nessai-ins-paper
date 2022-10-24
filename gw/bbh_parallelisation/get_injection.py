"""Get one of the injections from the JSON file and save it."""
import json


def main():

    injection_number = 20
    injection_file = "../bbh_pp_test/precessing_injections.json"

    with open(injection_file, "r") as fp:
        all_inj = json.load(fp)

    single_injection = all_inj.copy()
    for parameter, values in all_inj["injections"]["content"].items():
        single_injection["injections"]["content"][parameter] = [
            values[injection_number]
        ]

    print(f"Injection parameters: {single_injection['injections']['content']}")

    with open("parallelisation_injection.json", "w") as fp:
        json.dump(single_injection, fp)


if __name__ == "__main__":
    main()
